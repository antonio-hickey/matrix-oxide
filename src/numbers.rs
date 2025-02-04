/// Marker trait for categorizing the set of all number types
/// and enforcing type constraints.
pub trait Numeric: AsF64 {}
impl Numeric for i8 {}
impl Numeric for i16 {}
impl Numeric for i32 {}
impl Numeric for i64 {}
impl Numeric for i128 {}
impl Numeric for u8 {}
impl Numeric for u16 {}
impl Numeric for u32 {}
impl Numeric for u64 {}
impl Numeric for u128 {}
impl Numeric for f32 {}
impl Numeric for f64 {}

/// Marker trait for categorizing the set of integer
/// number types and enforcing type constraints.
pub trait Integers {}
impl Integers for i8 {}
impl Integers for i16 {}
impl Integers for i32 {}
impl Integers for i64 {}
impl Integers for i128 {}
impl Integers for u8 {}
impl Integers for u16 {}
impl Integers for u32 {}
impl Integers for u64 {}
impl Integers for u128 {}

/// Marker trait for categorizing the set of floating
/// point number types and enforcing type constraints.
pub trait Floats {}
impl Floats for f32 {}
impl Floats for f64 {}

/// Convert something to f64.
/// NOTE: This is lossy for big integers.
pub trait AsF64 {
    fn as_f64(&self) -> f64;
}
/// Convert an i8 to an f64.
impl AsF64 for i8 {
    fn as_f64(&self) -> f64 {
        *self as f64
    }
}
impl AsF64 for i16 {
    fn as_f64(&self) -> f64 {
        *self as f64
    }
}
impl AsF64 for i32 {
    fn as_f64(&self) -> f64 {
        *self as f64
    }
}
impl AsF64 for i64 {
    fn as_f64(&self) -> f64 {
        *self as f64
    }
}
impl AsF64 for i128 {
    fn as_f64(&self) -> f64 {
        *self as f64
    }
}
impl AsF64 for u8 {
    fn as_f64(&self) -> f64 {
        *self as f64
    }
}
impl AsF64 for u16 {
    fn as_f64(&self) -> f64 {
        *self as f64
    }
}
impl AsF64 for u32 {
    fn as_f64(&self) -> f64 {
        *self as f64
    }
}
impl AsF64 for u64 {
    fn as_f64(&self) -> f64 {
        *self as f64
    }
}
impl AsF64 for u128 {
    fn as_f64(&self) -> f64 {
        *self as f64
    }
}
impl AsF64 for f32 {
    fn as_f64(&self) -> f64 {
        *self as f64
    }
}
impl AsF64 for f64 {
    fn as_f64(&self) -> f64 {
        *self
    }
}
