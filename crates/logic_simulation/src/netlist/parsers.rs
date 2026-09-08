use crate::Signal;
use chumsky::{
    prelude::*,
    text::{newline, whitespace},
};
use std::ops::Range;

type Extra<'src> = extra::Err<Rich<'src, char>>;

enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

impl<T> OneOrMany<T> {
    fn map<F, U>(self, f: F) -> OneOrMany<U>
    where
        F: Fn(T) -> U,
    {
        match self {
            OneOrMany::One(item) => OneOrMany::One(f(item)),
            OneOrMany::Many(items) => OneOrMany::Many(items.into_iter().map(f).collect()),
        }
    }
}

/// A statement in the parsed module.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Statement<'src> {
    /// The identifier of the statement. For example the second token in `input
    /// a` or `NAND2_X1 abc`.
    pub name: &'src str,
    /// The span of the statement in the source code.
    pub span: SimpleSpan,
    /// Kind specific data.
    pub kind: StatementKind<'src>,
}

/// The kind of statement.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StatementKind<'src> {
    /// An input statement
    ///
    /// For example `input a`.
    Input,
    /// An input bus statement
    ///
    /// For example `input [31:0] a`.
    InputBus {
        /// The indices of the bus.
        range: Range<usize>,
    },
    /// An output statement
    ///
    /// For example `output z`.
    Output,
    /// An output bus statement
    ///
    /// For example `output [31:0] z`.
    OutputBus {
        /// The indices of the bus.
        range: Range<usize>,
    },
    /// A wire statement
    ///
    /// For example `wire a`.
    Wire,
    /// A bus statement
    ///
    /// For example `wire [31:0] a`.
    Bus {
        /// The indices of the bus.
        range: Range<usize>,
    },
    /// An assign statement
    ///
    /// For example `assign status[6] = 1'b0`.
    Assignment {
        /// The index if assigning to a bus.
        index: Option<usize>,
        /// The value to assign.
        value: Signal,
    },
    /// A cell.
    ///
    /// For example `MUX2_X1 cool_name(.A (a), .B (b[5]), .S (s), .Z (z))`
    Cell {
        /// The kind of a cell.
        name: &'src str,
        /// The span of source code that covers the cell name.
        name_span: SimpleSpan,
        /// The span of source code that covers port definitions.
        ports_span: SimpleSpan,
        /// The ports of the cell.
        ports: Box<[Port<'src>]>,
    },
}

type BoundsAndNames<'src> = (Option<(usize, usize)>, OneOrMany<&'src str>);

fn index<'src>() -> impl Parser<'src, &'src str, usize, Extra<'src>> {
    text::int(10)
        .try_map(|text: &str, span| {
            text.parse::<usize>()
                .map_err(|err| Rich::custom(span, format!("invalid index: {}", err)))
        })
        .labelled("index")
}

fn bounds_and_names<'src>(
    prefix: &'src str,
) -> impl Parser<'src, &'src str, BoundsAndNames<'src>, Extra<'src>> {
    let bus_bounds = index()
        .then_ignore(just(':').padded())
        .then(index())
        .padded()
        .delimited_by(just('['), just(']'))
        .then_ignore(whitespace().at_least(1))
        .or_not()
        .labelled("bus bounds");

    let names = identifier()
        .separated_by(just(',').padded())
        .at_least(2)
        .collect::<Vec<_>>()
        .map(OneOrMany::Many);
    let name = identifier().padded().map(OneOrMany::One);

    just(prefix)
        .ignore_then(whitespace().at_least(1))
        .ignore_then(bus_bounds)
        .then(names.or(name))
        .boxed()
}

fn statement<'src>() -> impl Parser<'src, &'src str, OneOrMany<Statement<'src>>, Extra<'src>> {
    let wire = bounds_and_names("wire")
        .map_with(|(bounds, names), extra| {
            let span = extra.span();
            names.map(|name| match bounds {
                Some((end, start)) => Statement {
                    name,
                    span,
                    kind: StatementKind::Bus {
                        range: start..(end + 1),
                    },
                },
                None => Statement {
                    name,
                    span,
                    kind: StatementKind::Wire,
                },
            })
        })
        .labelled("wire declaration");

    let input = bounds_and_names("input")
        .map_with(|(bounds, names), extra| {
            let span = extra.span();
            names.map(|name| match bounds {
                Some((end, start)) => Statement {
                    name,
                    span,
                    kind: StatementKind::InputBus {
                        range: start..(end + 1),
                    },
                },
                None => Statement {
                    name,
                    span,
                    kind: StatementKind::Input,
                },
            })
        })
        .labelled("input declaration");

    let output = bounds_and_names("output")
        .map_with(|(bounds, names), extra| {
            let span = extra.span();
            names.map(|name| match bounds {
                Some((end, start)) => Statement {
                    name,
                    span,
                    kind: StatementKind::OutputBus {
                        range: start..(end + 1),
                    },
                },
                None => Statement {
                    name,
                    span,
                    kind: StatementKind::Output,
                },
            })
        })
        .labelled("output declaration");

    let signal = choice((
        just('0').to(Signal::Low),
        just('1').to(Signal::High),
        just('x').or(just('X')).to(Signal::Unknown),
    ));

    let assign_index = index().padded().delimited_by(just('['), just(']'));
    let assign_value = just("1'b").ignore_then(signal);
    let assignment = just("assign")
        .labelled("keyword `assign`")
        .ignore_then(whitespace().at_least(1))
        .ignore_then(identifier())
        .then(assign_index.or_not().padded())
        .then_ignore(just('=').padded())
        .then(assign_value)
        .map_with(|((name, index), value), extra| {
            OneOrMany::One(Statement {
                name,
                span: extra.span(),
                kind: StatementKind::Assignment { index, value },
            })
        });

    let component = cell().map(OneOrMany::One);

    choice((wire, input, output, assignment, component))
        .padded()
        .then_ignore(just(';'))
        .boxed()
}

fn identifier<'src>() -> impl Parser<'src, &'src str, &'src str, Extra<'src>> {
    let ascii = any()
        .filter(|c: &char| c.is_ascii() && !c.is_whitespace())
        .labelled("non-whitespace ascii");

    let regular_text = any()
        .filter(|c: &char| c.is_ascii_alphanumeric() || *c == '_' || *c == '$')
        .labelled("a-z A-Z 0-9 _ $");

    let escaped = just('\\').then(ascii.repeated().at_least(1)).ignored();

    let non_escaped = regular_text.repeated().at_least(1).ignored();

    escaped.or(non_escaped).to_slice().boxed()
}

fn comment<'src>() -> impl Parser<'src, &'src str, &'src str, Extra<'src>> {
    just("//")
        .then(any().and_is(newline().not()).repeated())
        .then(newline().or(end()))
        .to_slice()
        .padded()
        .labelled("comment")
}

fn empty_lines<'src>() -> impl Parser<'src, &'src str, &'src str, Extra<'src>> {
    comment()
        .or(whitespace().at_least(1).to_slice())
        .repeated()
        .to_slice()
        .labelled("comment")
        .boxed()
}

/// A port on a cell.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Port<'src> {
    /// The span of the port declaration.
    pub span: SimpleSpan,
    /// The name of the port.
    pub name: &'src str,
    /// The target of the port.
    pub target: Target<'src>,
}

/// The target of a port.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Target<'src> {
    /// The name of the target.
    pub name: &'src str,
    /// The index in the target bus, if it is a bus. `None` for wires.
    pub index: Option<usize>,
}

fn target<'src>() -> impl Parser<'src, &'src str, Target<'src>, Extra<'src>> {
    identifier()
        .then(index().padded().delimited_by(just('['), just(']')).or_not())
        .map(|(name, index)| Target { name, index })
        .padded()
        .delimited_by(just('('), just(')'))
}

fn port<'src>() -> impl Parser<'src, &'src str, Port<'src>, Extra<'src>> {
    just('.')
        .ignore_then(identifier())
        .then_ignore(whitespace().at_least(1))
        .then(target())
        .map_with(|(name, target), extra| Port {
            span: extra.span(),
            name,
            target,
        })
}

fn port_map<'src>() -> impl Parser<'src, &'src str, (Box<[Port<'src>]>, SimpleSpan), Extra<'src>> {
    port()
        .separated_by(just(',').padded())
        .at_least(1)
        .collect()
        .padded()
        .delimited_by(just('('), just(')'))
        .labelled("port map")
        .map_with(|ports, extra| (ports, extra.span()))
        .boxed()
}

fn cell<'src>() -> impl Parser<'src, &'src str, Statement<'src>, Extra<'src>> {
    let prefix = any()
        .and_is(just('_').not())
        .repeated()
        .at_least(1)
        .to_slice()
        .then_ignore(just("_X").then(text::int(10).repeated()))
        .map_with(|name, extra| (name, extra.span()));

    prefix
        .then_ignore(whitespace().at_least(1))
        .then(identifier())
        .then(port_map().padded())
        .map_with(
            |(((cell_name, cell_name_span), name), (ports, ports_span)), extra| Statement {
                name,
                kind: StatementKind::Cell {
                    name: cell_name,
                    name_span: cell_name_span,
                    ports_span,
                    ports,
                },
                span: extra.span(),
            },
        )
        .labelled("cell declaration")
        .boxed()
}

/// A chumsky parser for the netlist.
pub fn parser<'src>() -> impl Parser<'src, &'src str, Vec<Statement<'src>>, Extra<'src>> {
    let statements = statement()
        .separated_by(empty_lines())
        .fold(Vec::new(), |mut acc, decl| {
            match decl {
                OneOrMany::One(item) => acc.push(item),
                OneOrMany::Many(items) => acc.extend(items),
            };
            acc
        });

    let module_args = none_of(')')
        .repeated()
        .labelled("module argument list")
        .delimited_by(just('('), just(')'))
        .labelled("module argument list");

    let module_header = just("module")
        .then(whitespace().at_least(1))
        .then(identifier())
        .then(module_args)
        .then(just(';'))
        .labelled("module header");

    let module = module_header
        .ignore_then(statements.padded_by(empty_lines()))
        .then_ignore(just("endmodule"))
        .then_ignore(empty_lines().or_not())
        .then_ignore(end());

    empty_lines().ignore_then(module).then_ignore(empty_lines())
}
