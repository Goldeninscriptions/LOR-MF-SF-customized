#ifndef LOR_ASSEMBLY_PRECONDITIONER_SIDE_HPP
#define LOR_ASSEMBLY_PRECONDITIONER_SIDE_HPP

namespace lor
{

// Algebraic side on which the LOR inverse acts. Left represents
// K_LOR^{-1} K u = K_LOR^{-1} F; right represents
// K K_LOR^{-1} y = F followed by u = K_LOR^{-1} y.
enum class PreconditionerSide
{
    left,
    right
};

} // namespace lor

#endif
