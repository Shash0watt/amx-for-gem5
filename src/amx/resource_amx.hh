#ifndef __AMX_RESOURCE_AMX_HH__
#define __AMX_RESOURCE_AMX_HH__

#include <array>
#include <cstddef>
#include <cstdint>

#include "base/types.hh"

namespace gem5
{
namespace amx
{

// Independently pipelined AMX issue resources. TileZero and TileStore are
// included now so their future execution paths use the same timing model.
enum class Resource
{
    TileLoad,
    DotProduct,
    TileZero,
    TileStore,
    Count
};

struct ResourceState
{
    // Minimum cycles between instructions issued to this pipeline.
    Cycles initiationInterval = Cycles(0);

    // First cycle when this pipeline can accept another instruction.
    Cycles nextIssueCycle = Cycles(0);

    // Instructions issued to this pipeline that have not completed yet.
    unsigned inFlight = 0;

    // Total instructions issued to this pipeline.
    uint64_t issueCount = 0;
};

class ResourceTracker
{
  public:
    ResourceTracker(Cycles load_interval, Cycles dot_product_interval,
                    Cycles zero_interval, Cycles store_interval);

    bool canIssue(Resource resource, Cycles now) const;
    Cycles nextIssueCycle(Resource resource) const;
    void issue(Resource resource, Cycles now);
    void complete(Resource resource);

    const ResourceState &state(Resource resource) const;

  private:
    static constexpr size_t NumResources =
        static_cast<size_t>(Resource::Count);

    ResourceState &mutableState(Resource resource);
    std::array<ResourceState, NumResources> states;
};

const char *resourceName(Resource resource);

} // namespace amx
} // namespace gem5

#endif // __AMX_RESOURCE_AMX_HH__
