// Subgroup-accelerated workgroup reduction helpers.
//
// Replaces the traditional shared-memory tree reduction (log2(N) barriers)
// with a two-phase approach:
//   Phase 1: subgroupAdd/subgroupMax within each subgroup (0 barriers)
//   Phase 2: one barrier + first subgroup reduces partial results
//
// These are macros because GLSL shared arrays cannot be passed as function
// parameters. The caller must provide the shared scratch array name.
//
// Requires: GL_KHR_shader_subgroup_basic, GL_KHR_shader_subgroup_arithmetic

#ifndef SUBGROUP_REDUCE_GLSL
#define SUBGROUP_REDUCE_GLSL

// SUBGROUP_REDUCE_ADD(val, scratch)
//   val:     FLOAT_TYPE variable to reduce (modified in-place with result)
//   scratch: name of a shared FLOAT_TYPE array with at least BLOCK_SIZE elements
//   Result is broadcast to ALL invocations via val.
#define SUBGROUP_REDUCE_ADD(val, scratch) \
    { \
        val = subgroupAdd(val); \
        if (subgroupElect()) { \
            scratch[gl_SubgroupID] = val; \
        } \
        barrier(); \
        if (gl_SubgroupID == 0u) { \
            val = (gl_SubgroupInvocationID < gl_NumSubgroups) \
                ? scratch[gl_SubgroupInvocationID] \
                : FLOAT_TYPE(0.0f); \
            val = subgroupAdd(val); \
            scratch[0] = val; \
        } \
        barrier(); \
        val = scratch[0]; \
    }

// SUBGROUP_REDUCE_MAX(val, scratch)
//   val:     FLOAT_TYPE variable to reduce (modified in-place with result)
//   scratch: name of a shared FLOAT_TYPE array with at least BLOCK_SIZE elements
//   Result is broadcast to ALL invocations via val.
#define SUBGROUP_REDUCE_MAX(val, scratch) \
    { \
        val = subgroupMax(val); \
        if (subgroupElect()) { \
            scratch[gl_SubgroupID] = val; \
        } \
        barrier(); \
        if (gl_SubgroupID == 0u) { \
            val = (gl_SubgroupInvocationID < gl_NumSubgroups) \
                ? scratch[gl_SubgroupInvocationID] \
                : uintBitsToFloat(0xFF800000); \
            val = subgroupMax(val); \
            scratch[0] = val; \
        } \
        barrier(); \
        val = scratch[0]; \
    }

// Float (non-FLOAT_TYPE) variants for shaders that use float directly
#define SUBGROUP_REDUCE_ADD_F(val, scratch) \
    { \
        val = subgroupAdd(val); \
        if (subgroupElect()) { \
            scratch[gl_SubgroupID] = val; \
        } \
        barrier(); \
        if (gl_SubgroupID == 0u) { \
            val = (gl_SubgroupInvocationID < gl_NumSubgroups) \
                ? scratch[gl_SubgroupInvocationID] \
                : 0.0f; \
            val = subgroupAdd(val); \
            scratch[0] = val; \
        } \
        barrier(); \
        val = scratch[0]; \
    }

#endif // SUBGROUP_REDUCE_GLSL
