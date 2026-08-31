# `strata.io`

Zero-copy, endian-safe binary serialization engine (dump, load, dumps, loads) for fitted estimators, transformers, trees, and ensembles.

---

## Structs & Classes

| Struct | Description |
| :--- | :--- |
| [`BufferWriter`](BufferWriter.md) | Byte buffer writer with endian-safe scalar and matrix serialization. |
| [`BufferReader`](BufferReader.md) | Byte buffer reader with bounds checking and scalar/matrix deserialization. |

## Traits

| Trait | Description |
| :--- | :--- |
| [`Serializable`](Serializable.md) | Trait for models and transformers supporting binary serialization. |
