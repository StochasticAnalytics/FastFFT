#ifndef _SRC_FASTFFT_TYPES_H_
#define _SRC_FASTFFT_TYPES_H_

#include <array>
#include <string_view>

namespace FastFFT {

namespace DataType {
// Used to specify input/calc/output data types
enum Enum { int4_2,
            uint8,
            int8,
            uint16,
            int16,
            fp16,
            bf16,
            tf32,
            uint32,
            int32,
            fp32 };

constexpr std::array<std::string_view, 11> name = {"int4_2", "uint8", "int8", "uint16", "int16", "fp16", "bf16", "tf32", "uint32", "int32", "fp32"};

} // namespace DataType

namespace SizeChangeType {
// FIXME this seems like a bad idea. Added due to conflicing labels in switch statements, even with explicitly scope.
enum Enum : uint8_t { increase,
                      decrease,
                      no_change };
} // namespace SizeChangeType

namespace OriginType {
// Used to specify the origin of the data
enum Enum : int { natural,
                  centered,
                  quadrant_swapped };

constexpr std::array<std::string_view, 3> name = {"natural", "centered", "quadrant_swapped"};

} // namespace OriginType

namespace TransformStageCompleted {
enum Enum : uint8_t { none = 10,
                      fwd  = 11,
                      inv  = 12 }; // none must be greater than number of sizeChangeTypes, padding must match in TransformStageCompletedName vector
} // namespace TransformStageCompleted

namespace DimensionCheckType {
enum Enum : uint8_t { CopyFromHost,
                      CopyToHost,
                      FwdTransform,
                      InvTransform };

} // namespace DimensionCheckType

} // namespace FastFFT

#endif /* _SRC_FASTFFT_TYPES_H_ */