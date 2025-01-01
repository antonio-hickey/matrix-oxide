Matrix Oxide
===

A simple, lightweight, and from scratch linear algebra library for Rust. Currently still under active development with goals at becoming more of a deep learning library.

__This is currently a pre-release version__

Installation
---
Use cargo CLI:
```
cargo install matrix-oxide
```

Or manually add it into your Cargo.toml:
```
[dependencies]
matrix-oxide = "0.0.1"
```

Usage
---

For more thorough information, read the [docs](https://docs.rs/matrix-oxide/latest/matrix-oxide/index.html).


Example: Multiply 2 random 2x2 matrices.

```
use matrix_oxide::Matrix;

fn main() {
    let matrix_a = Matrix::<i32>::new_random(2, 2);
    let matrix_b = Matrix::<i32>::new_random(2, 2);

    matrix_a.multiply(&matrix_b).unwrap();
}
```
