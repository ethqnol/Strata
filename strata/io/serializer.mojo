from std.memory import unsafe_memcpy, bitcast
from std.sys.info import size_of
from ..core.matrix import Matrix
from ..exceptions.errors import InvalidParameterError, DataConversionError


struct BufferWriter:
    """Byte buffer writer with endian-safe scalar and matrix serialization."""

    var buffer: List[UInt8]

    def __init__(out self):
        self.buffer = List[UInt8]()

    def write_byte(mut self, val: UInt8):
        self.buffer.append(val)

    def write_bool(mut self, val: Bool):
        self.buffer.append(UInt8(1) if val else UInt8(0))

    def write_int(mut self, val: Int):
        var v = UInt64(val)
        for i in range(8):
            var b = (v >> UInt64(i * 8)) & 0xFF
            self.buffer.append(b.cast[DType.uint8]())

    def write_float64(mut self, val: Float64):
        var v = bitcast[DType.uint64](val)
        for i in range(8):
            var b = (v >> UInt64(i * 8)) & 0xFF
            self.buffer.append(b.cast[DType.uint8]())

    def write_float32(mut self, val: Float32):
        var v = bitcast[DType.uint32](val)
        for i in range(4):
            var b = (v >> UInt32(i * 8)) & 0xFF
            self.buffer.append(b.cast[DType.uint8]())

    def write_string(mut self, val: String):
        var b = val.as_bytes()
        self.write_int(len(b))
        for i in range(len(b)):
            self.buffer.append(b[i])

    def write_int_list(mut self, val: List[Int]):
        self.write_int(len(val))
        for i in range(len(val)):
            self.write_int(val[i])

    def write_float64_list(mut self, val: List[Float64]):
        self.write_int(len(val))
        for i in range(len(val)):
            self.write_float64(val[i])

    def write_float_list[dtype: DType](mut self, val: List[Scalar[dtype]]):
        self.write_int(len(val))
        for i in range(len(val)):
            self.write_float64(val[i].cast[DType.float64]())

    def write_matrix[dtype: DType](mut self, val: Matrix[dtype]):
        self.write_int(val.rows)
        self.write_int(val.cols)
        var num_elem = val.rows * val.cols
        if num_elem > 0:
            var num_bytes = num_elem * size_of[dtype]()
            var cur_len = len(self.buffer)
            self.buffer.resize(cur_len + num_bytes, 0)
            var src_ptr = val.data.unsafe_ptr().unsafe_bitcast[UInt8]()
            var dst_ptr = self.buffer.unsafe_ptr().unsafe_offset(cur_len)
            unsafe_memcpy(dest=dst_ptr, src=src_ptr, count=num_bytes)

    def get_bytes(self) -> List[UInt8]:
        return self.buffer.copy()

    def save_to_file(self, path: String) raises:
        var f = open(path, "w")
        f.write_bytes(Span(self.buffer))
        f.close()


struct BufferReader:
    """Byte buffer reader with bounds checking and scalar/matrix deserialization.
    """

    var buffer: List[UInt8]
    var cursor: Int

    def __init__(out self, buffer: List[UInt8]):
        self.buffer = buffer.copy()
        self.cursor = 0

    @staticmethod
    def from_file(path: String) raises -> Self:
        var f = open(path, "r")
        var bytes = f.read_bytes()
        f.close()
        return BufferReader(bytes^)

    def read_byte(mut self) raises -> UInt8:
        if self.cursor >= len(self.buffer):
            raise InvalidParameterError.error(
                "BufferReader", "Unexpected EOF reading byte"
            )
        var b = self.buffer[self.cursor]
        self.cursor += 1
        return b

    def read_bool(mut self) raises -> Bool:
        return self.read_byte() != 0

    def read_int(mut self) raises -> Int:
        if self.cursor + 8 > len(self.buffer):
            raise InvalidParameterError.error(
                "BufferReader", "Unexpected EOF reading Int"
            )
        var v: UInt64 = 0
        for i in range(8):
            var b = UInt64(self.buffer[self.cursor + i])
            v |= b << UInt64(i * 8)
        self.cursor += 8
        return Int(v)

    def read_float64(mut self) raises -> Float64:
        if self.cursor + 8 > len(self.buffer):
            raise InvalidParameterError.error(
                "BufferReader", "Unexpected EOF reading Float64"
            )
        var v: UInt64 = 0
        for i in range(8):
            var b = UInt64(self.buffer[self.cursor + i])
            v |= b << UInt64(i * 8)
        self.cursor += 8
        return bitcast[DType.float64](v)

    def read_float32(mut self) raises -> Float32:
        if self.cursor + 4 > len(self.buffer):
            raise InvalidParameterError.error(
                "BufferReader", "Unexpected EOF reading Float32"
            )
        var v: UInt32 = 0
        for i in range(4):
            var b = UInt32(self.buffer[self.cursor + i])
            v |= b << UInt32(i * 8)
        self.cursor += 4
        return bitcast[DType.float32](v)

    def read_string(mut self) raises -> String:
        var str_len = self.read_int()
        if self.cursor + str_len > len(self.buffer):
            raise InvalidParameterError.error(
                "BufferReader", "Unexpected EOF reading String"
            )
        var b = List[UInt8](capacity=str_len)
        for i in range(str_len):
            b.append(self.buffer[self.cursor + i])
        self.cursor += str_len
        return String(unsafe_from_utf8=b)

    def read_int_list(mut self) raises -> List[Int]:
        var list_len = self.read_int()
        var res = List[Int](capacity=list_len)
        for _ in range(list_len):
            res.append(self.read_int())
        return res^

    def read_float64_list(mut self) raises -> List[Float64]:
        var list_len = self.read_int()
        var res = List[Float64](capacity=list_len)
        for _ in range(list_len):
            res.append(self.read_float64())
        return res^

    def read_float_list[dtype: DType](mut self) raises -> List[Scalar[dtype]]:
        var list_len = self.read_int()
        var res = List[Scalar[dtype]](capacity=list_len)
        for _ in range(list_len):
            res.append(Scalar[dtype](self.read_float64()))
        return res^

    def read_matrix[dtype: DType](mut self) raises -> Matrix[dtype]:
        var rows = self.read_int()
        var cols = self.read_int()
        var m = Matrix[dtype](rows, cols, 0)
        var num_elem = rows * cols
        if num_elem > 0:
            var num_bytes = num_elem * size_of[dtype]()
            if self.cursor + num_bytes > len(self.buffer):
                raise InvalidParameterError.error(
                    "BufferReader", "Unexpected EOF reading Matrix"
                )
            var src_ptr = self.buffer.unsafe_ptr().unsafe_offset(self.cursor)
            var dst_ptr = m.data.unsafe_ptr().unsafe_bitcast[UInt8]()
            unsafe_memcpy(dest=dst_ptr, src=src_ptr, count=num_bytes)
            self.cursor += num_bytes
        return m^


# =========================================================================
# Magic Header & Protocol Constants
# =========================================================================

comptime FORMAT_MAGIC = "STRATA"
comptime FORMAT_VERSION = 1


def write_header(mut writer: BufferWriter, type_name: String):
    """Writes protocol magic header, version tag, and model type identifier."""
    writer.write_string(FORMAT_MAGIC)
    writer.write_int(FORMAT_VERSION)
    writer.write_string(type_name)


def check_header(mut reader: BufferReader, expected_type: String) raises:
    """Validates protocol magic header, version tag, and model type identifier.
    """
    var magic = reader.read_string()
    if magic != FORMAT_MAGIC:
        raise DataConversionError.error(
            "Invalid file format: magic header mismatch (expected '"
            + FORMAT_MAGIC
            + "', got '"
            + magic
            + "')"
        )
    var version = reader.read_int()
    if version != FORMAT_VERSION:
        raise DataConversionError.error(
            "Unsupported format version "
            + String(version)
            + " (expected "
            + String(FORMAT_VERSION)
            + ")"
        )
    var type_name = reader.read_string()
    if type_name != expected_type:
        raise DataConversionError.error(
            "Model type mismatch: file contains '"
            + type_name
            + "', expected '"
            + expected_type
            + "'"
        )


# =========================================================================
# Serializable Trait & Top-Level Persistence API
# =========================================================================


trait Serializable(Copyable, Movable):
    """Trait for models and transformers supporting binary serialization."""

    def serialize(self, mut writer: BufferWriter):
        """Serializes internal parameters and fitted state into BufferWriter."""
        ...

    @staticmethod
    def deserialize(mut reader: BufferReader) raises -> Self:
        """Deserializes a model instance from BufferReader."""
        ...


def dump[T: Serializable](model: T, path: String) raises:
    """Serializes a fitted estimator or transformer to disk.

    Args:
        model: Fitted estimator or transformer instance.
        path: Destination filepath.
    """
    var writer = BufferWriter()
    model.serialize(writer)
    writer.save_to_file(path)


def dumps[T: Serializable](model: T) raises -> List[UInt8]:
    """Serializes a fitted estimator or transformer into an in-memory byte buffer.

    Args:
        model: Fitted estimator or transformer instance.

    Returns:
        List[UInt8]: Packed binary byte buffer.
    """
    var writer = BufferWriter()
    model.serialize(writer)
    return writer.get_bytes()


def load[T: Serializable](path: String) raises -> T:
    """Deserializes a fitted estimator or transformer from disk.

    Parameters:
        T: Concrete model struct type.

    Args:
        path: Filepath of the serialized model binary.

    Returns:
        T: Restored model instance with fitted state intact.
    """
    var reader = BufferReader.from_file(path)
    return T.deserialize(reader)


def loads[T: Serializable](bytes: List[UInt8]) raises -> T:
    """Deserializes a fitted estimator or transformer from an in-memory byte buffer.

    Parameters:
        T: Concrete model struct type.

    Args:
        bytes: Packed binary byte buffer.

    Returns:
        T: Restored model instance with fitted state intact.
    """
    var reader = BufferReader(bytes)
    return T.deserialize(reader)
