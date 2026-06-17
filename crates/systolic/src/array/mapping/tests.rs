use super::*;
// --- Valid mappings ---

/// A single pass that covers all input-row/output-row combinations is valid.
#[test]
fn validate_single_pass_complete() {
    assert_eq!(Mapping::new([Pass::new(0..2, 0..2)]).validate(), Ok(()));
}

/// Multiple passes that together cover all combinations without overlap is valid.
#[test]
fn validate_multiple_passes_complete() {
    // Pass 0: input rows {0,1} x output rows {0}
    // Pass 1: input rows {0,1} x output rows {1}
    assert_eq!(
        Mapping::new([Pass::new(0..2, 0..1), Pass::new(0..2, 1..2)]).validate(),
        Ok(())
    );
}

#[test]
fn validate_single_output() {
    assert_eq!(Mapping::new([Pass::new(0..1, 0..1)]).validate(), Ok(()));

    assert_eq!(Mapping::new([Pass::new(0..2, 0..1)]).validate(), Ok(()));
}

#[test]
fn validate_output_split_across_passes() {
    assert_eq!(
        Mapping::new([Pass::new(0..2, 0..1), Pass::new(2..4, 0..1)]).validate(),
        Ok(())
    );
}

// --- No connections ---

/// An empty passes list has no connections at all.
#[test]
#[should_panic]
fn no_passes() {
    Mapping::new([]);
}

// --- Duplicate connections ---

/// The same input-row/output-row pair appearing in two separate passes is a duplicate.
#[test]
fn validate_duplicate_connection_across_passes() {
    // Both passes contain input row 0 and output row 0, so connection 0->0 is duplicated.
    assert!(matches!(
        Mapping::new([Pass::new(0..1, 0..1), Pass::new(0..1, 0..1)]).validate(),
        Err(InvalidMappingError::DuplicateConnection { .. })
    ));
}

/// A pass that partially overlaps with another pass causes a duplicate.
#[test]
fn validate_partial_overlap_causes_duplicate() {
    // Pass 0 covers input rows {0,1} x output rows {0,1} (all combos).
    // Pass 1 adds input row 0 x output row 1 again -> duplicate.
    assert!(matches!(
        Mapping::new([Pass::new(0..2, 0..2), Pass::new(0..1, 1..2)]).validate(),
        Err(InvalidMappingError::DuplicateConnection {
            activation_row: 0,
            output_row: 1
        })
    ));
}

// --- Missing connections ---

/// When only some input-row/output-row combinations are covered, the rest are missing.
#[test]
fn validate_missing_connections() {
    // Pass 0 connects row 0->0 and pass 1 connects row 1->1,
    // leaving connections 0->1 and 1->0 missing.
    let missing = [Connection::new(0, 1), Connection::new(1, 0)];
    assert_eq!(
        Mapping::new([Pass::new(0..1, 0..1), Pass::new(1..2, 1..2)]).validate(),
        Err(InvalidMappingError::MissingConnections(missing.into()))
    );
}

/// A mapping that covers rows split across passes but misses a column is invalid.
#[test]
fn validate_split_passes_missing_column() {
    // Input rows go up to index 1, output rows go up to index 1.
    // Pass 0 covers input rows {0,1} x output rows {0} and pass 1 covers
    // input row {0} x output row {1}, leaving connection 1->1 missing.
    let missing = [Connection::new(1, 1)];

    assert_eq!(
        Mapping::new([Pass::new(0..2, 0..1), Pass::new(0..1, 1..2)]).validate(),
        Err(InvalidMappingError::MissingConnections(missing.into()))
    );
}

#[test]
fn test_range_get() {
    let range = 0..2;
    assert_eq!(range_get(&range, 0), Some(0));
    assert_eq!(range_get(&range, 1), Some(1));
    assert_eq!(range_get(&range, 2), None);
}

#[test]
fn array_index_to_io() {
    let pass = Pass::new(0..2, 0..3);
    assert_eq!(pass.activation_row_from_array_row(0), Some(0));
    assert_eq!(pass.activation_row_from_array_row(1), Some(1));
    assert_eq!(pass.output_row_from_array_col(0), Some(0));
    assert_eq!(pass.output_row_from_array_col(1), Some(1));

    let pass_with_offset = Pass::new(0..2, 0..3).with_offset(1, 1);
    assert_eq!(pass_with_offset.activation_row_from_array_row(0), None);
    assert_eq!(pass_with_offset.activation_row_from_array_row(1), Some(0));
    assert_eq!(pass_with_offset.activation_row_from_array_row(2), Some(1));
    assert_eq!(pass_with_offset.activation_row_from_array_row(3), None);
    assert_eq!(pass_with_offset.output_row_from_array_col(0), None);
    assert_eq!(pass_with_offset.output_row_from_array_col(1), Some(0));
    assert_eq!(pass_with_offset.output_row_from_array_col(2), Some(1));
    assert_eq!(pass_with_offset.output_row_from_array_col(3), Some(2));
    assert_eq!(pass_with_offset.output_row_from_array_col(4), None);
}

#[test]
fn array_ranges() {
    let pass = Pass::new(0..2, 0..3);
    assert_eq!(pass.range_y(), 0..2);
    assert_eq!(pass.range_x(), 0..3);

    let pass = Pass::new(0..2, 0..3).with_offset(1, 1);
    assert_eq!(pass.range_y(), 1..3);
    assert_eq!(pass.range_x(), 1..4);
}

#[test]
fn inconsistent_passes() {
    let mapping = Mapping::new([
        Pass::new(0..2, 0..1),
        Pass::new(0..2, 1..2).with_offset(2, 0),
    ]);
    assert_eq!(
        mapping.validate(),
        Err(InvalidMappingError::InconsistentActivation {
            activation_row: 0,
            array_row1: 0,
            array_row1_pass: 0,
            array_row2: 2,
            array_row2_pass: 1
        })
    );

    let mapping = Mapping::new([
        Pass::new(0..2, 0..1).with_offset(0, 1),
        Pass::new(0..2, 1..2).with_offset(0, 1),
        Pass::new(2..3, 0..2).with_offset(0, 3),
    ]);
    assert_eq!(
        mapping.validate(),
        Err(InvalidMappingError::InconsistentOutput {
            output_row: 0,
            array_col1: 1,
            array_col1_pass: 0,
            array_col2: 3,
            array_col2_pass: 2
        })
    );
}
