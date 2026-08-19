use crate::{common::*, fault::PyFault};
use memory::{
    BitBuffer, ByteBuffer, Limited, SizedBitBuffer,
    chunks::{Chunks, ChunksCreationError, DecodeError, DynChunks},
    encoding::secded::encoded_bit_count,
    sequence::NonUniformSequence,
};
use numpy::PyArray1;
use pyo3::{
    exceptions::{PyIndexError, PyValueError},
    prelude::*,
    types::{PyBytes, PyList},
};

#[pyclass(name = "Encoding")]
pub struct PyEncoding {
    /// A list of the encoded chunks. Guaranteed to store [`PyBytes`] elements.
    encoded_chunks: Py<PyList>,
    /// The number of data bits per chunk, used for bounds in `Limited`.
    data_bit_count: usize,
    /// The number of items in each array in the input list for the encoding
    /// function.
    item_counts: Vec<usize>,
}

impl PyEncoding {
    pub fn decode_generic<'py, T>(
        &self,
        py: Python<'py>,
        mut output_buffer: NonUniformSequence<Vec<Vec<T>>>,
    ) -> PyResult<(Vec<OutputArr<'py, T>>, Vec<bool>)>
    where
        T: SizedBitBuffer + numpy::Element + ByteBuffer,
    {
        let input_chunks = self.to_rust(py)?;

        let (output_chunks, decoding_results) = input_chunks
            .decode_chunks(self.data_bit_count)
            .unwrap_or_else(|err| match err {
                DecodeError::InvalidDataBitsCount(_) => {
                    panic!("The data bits count is immutable from the python side and should be correct yet: {}", err);
                }
            });

        let bits_copied = match output_chunks {
            Chunks::Byte(byte_chunks) => {
                byte_chunks
                    .copy_into_chunked(&mut output_buffer)
                    .units_copied
                    * 8
            }
            Chunks::Dyn(dyn_chunks) => dyn_chunks.copy_into(&mut output_buffer).units_copied,
        };
        assert_eq!(
            bits_copied,
            output_buffer.bit_count(),
            "these must match because the chunked buffer can potentially only have more bits, not less.",
        );

        Ok((
            output_buffer
                .0
                .into_iter()
                .map(|vec| PyArray1::from_vec(py, vec))
                .collect(),
            decoding_results,
        ))
    }

    pub fn from_rust<'py>(
        py: Python<'py>,
        encoded_chunks: DynChunks,
        data_bit_count: usize,
        item_counts: Vec<usize>,
    ) -> PyResult<Self> {
        let bytes = encoded_chunks.into_raw().into_iter().map(|chunk| {
            assert_eq!(
                chunk.bit_count(),
                encoded_bit_count(data_bit_count)
                    .expect("chunk creation will fail before this for empty buffers.")
            );
            let chunk_bytes = chunk.into_inner();
            PyBytes::new(py, &chunk_bytes).unbind()
        });

        let encoded_chunks = PyList::new(py, bytes)?.unbind();

        Ok(PyEncoding {
            encoded_chunks,
            data_bit_count,
            item_counts,
        })
    }

    pub fn to_rust<'py>(&self, py: Python<'py>) -> PyResult<DynChunks> {
        let encoded_bit_count = encoded_bit_count(self.data_bit_count)
            .expect("Known to be correct. Checked during initialization.");

        let raw_chunks = self
            .encoded_chunks
            .bind(py)
            .iter()
            .map(|chunk| {
                let py_bytes = chunk.cast_into::<PyBytes>().expect(
                    "It was created as `PyBytes` and it should not be possible to modify it after.",
                );

                Limited::new(Vec::from(py_bytes.as_bytes()), encoded_bit_count).unwrap_or_else(
                    || {
                        panic!(
                            "`bits_per_chunk` {} doesn't match the encoded chunks.",
                            encoded_bit_count
                        )
                    },
                )
            })
            .collect::<Vec<_>>();

        Ok(
            DynChunks::from_raw(raw_chunks).expect(
                "There's no reason recreating the chunks should fail assuming the data remains immutable."
            )
        )
    }
}

#[pymethods]
impl PyEncoding {
    /// Decode a list of float32 values.
    pub fn decode_f32<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Vec<OutputArr<'py, f32>>, Vec<bool>)> {
        let output_buffer = NonUniformSequence(
            self.item_counts
                .iter()
                .map(|&numel| vec![0f32; numel])
                .collect::<Vec<_>>(),
        );

        self.decode_generic(py, output_buffer)
    }

    pub fn decode_u16<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Vec<OutputArr<'py, u16>>, Vec<bool>)> {
        let output_buffer = NonUniformSequence(
            self.item_counts
                .iter()
                .map(|&numel| vec![0u16; numel])
                .collect::<Vec<_>>(),
        );

        self.decode_generic(py, output_buffer)
    }

    pub fn apply_fault<'py>(
        &mut self,
        py: Python<'py>,
        fault: PyFault,
        target_bit: usize,
    ) -> PyResult<()> {
        self.apply_faults(py, vec![(fault, target_bit)])
    }

    /// Apply multiple faults at once.
    ///
    /// This is much cheaper than calling `apply_fault` in a loop: the whole
    /// encoded buffer is round-tripped through Python objects (`to_rust`/
    /// `from_rust`) exactly once for the batch instead of once per fault,
    /// even though flipping a single bit is itself O(1).
    pub fn apply_faults<'py>(
        &mut self,
        py: Python<'py>,
        faults: Vec<(PyFault, usize)>,
    ) -> PyResult<()> {
        let bit_count = self.bit_count(py);
        for (_, target_bit) in &faults {
            if *target_bit >= bit_count {
                return Err(PyIndexError::new_err(format!(
                    "target_bit {} is out of bounds",
                    target_bit
                )));
            }
        }

        let mut encoding = self.to_rust(py)?;
        encoding.apply_faults(
            faults
                .into_iter()
                .map(|(fault, target_bit)| (fault.0, target_bit)),
        );

        *self = PyEncoding::from_rust(py, encoding, self.data_bit_count, self.item_counts.clone())?;

        Ok(())
    }

    /// Return a new instance with cloned data.
    pub fn clone<'py>(&self, py: Python<'py>) -> PyResult<PyEncoding> {
        let encoding_clone = self.encoded_chunks.bind(py).iter().map(|chunk| {
            let bytes = chunk
                .cast_into::<PyBytes>()
                .expect("Was constructed as PyBytes and is immutable.");

            let cloned = bytes.as_bytes().to_owned();

            PyBytes::new(py, &cloned).unbind()
        });

        Ok(PyEncoding {
            encoded_chunks: PyList::new(py, encoding_clone)?.unbind(),
            data_bit_count: self.data_bit_count,
            item_counts: self.item_counts.clone(),
        })
    }

    pub fn bit_count<'py>(&self, py: Python<'py>) -> usize {
        self.encoded_chunks.bind(py).len()
            * encoded_bit_count(self.data_bit_count)
                .expect("the data bits count must be checked during initialization")
    }
}

fn encode_generic<'py, T>(
    py: Python<'py>,
    buffer: NonUniformSequence<Vec<Vec<T>>>,
    bits_per_chunk: usize,
) -> PyResult<PyEncoding>
where
    T: ByteBuffer + SizedBitBuffer,
{
    let item_counts = buffer.0.iter().map(|chunk| chunk.len()).collect::<Vec<_>>();

    let encoded_chunks = buffer
        .to_chunks(bits_per_chunk)
        .map_err(|err| match err {
            ChunksCreationError::Empty | ChunksCreationError::ZeroChunksize => {
                PyValueError::new_err(err.to_string())
            }
        })?
        .encode_chunks();

    PyEncoding::from_rust(py, encoded_chunks, bits_per_chunk, item_counts)
}

/// Encode a all bits of a buffer of 32 bit floats.
///
/// Returns a tuple containing:
/// - A uint8 array with the encoded bits
/// - The number of useful bits in the encoded buffer
///
/// These need to be given to the decoding function to restore the original representation.
#[pyfunction]
pub fn encode_f32<'py>(
    py: Python<'py>,
    input: Vec<InputArr<f32>>,
    bits_per_chunk: usize,
) -> PyResult<PyEncoding> {
    let buffer = prep_input_array_list(input);

    encode_generic(py, buffer, bits_per_chunk)
}

/// Encode a all bits of a buffer of 16 bit unsigned integers.
///
/// Returns a tuple containing:
/// - A uint8 array with the encoded bits
/// - The number of useful bits in the encoded buffer
///
/// These need to be given to the decoding function to restore the original representation.
#[pyfunction]
pub fn encode_u16<'py>(
    py: Python<'py>,
    input: Vec<InputArr<u16>>,
    bits_per_chunk: usize,
) -> PyResult<PyEncoding> {
    let buffer = prep_input_array_list(input);

    encode_generic(py, buffer, bits_per_chunk)
}
