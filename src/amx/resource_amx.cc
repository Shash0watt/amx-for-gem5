#include "amx/resource_amx.hh"

#include "base/logging.hh"

namespace gem5
{
namespace amx
{

namespace
{

size_t
resourceIndex(Resource resource)
{
    // Safely convert a resource name into its array position.
    const size_t index = static_cast<size_t>(resource);
    panic_if(index >= static_cast<size_t>(Resource::Count),
             "AMX resource index is invalid");
    return index;
}

} // anonymous namespace

ResourceTracker::ResourceTracker(Cycles load_interval,
                                 Cycles dot_product_interval,
                                 Cycles zero_interval,
                                 Cycles store_interval)
{
    // Give each independent pipeline its configured issue spacing.
    states[resourceIndex(Resource::TileLoad)].initiationInterval =
        load_interval;
    states[resourceIndex(Resource::DotProduct)].initiationInterval =
        dot_product_interval;
    states[resourceIndex(Resource::TileZero)].initiationInterval =
        zero_interval;
    states[resourceIndex(Resource::TileStore)].initiationInterval =
        store_interval;
}

bool
ResourceTracker::canIssue(Resource resource, Cycles now) const
{
    // The pipeline is ready once its minimum issue spacing has passed.
    return now >= state(resource).nextIssueCycle;
}

Cycles
ResourceTracker::nextIssueCycle(Resource resource) const
{
    // Tell the scheduler when it should try this pipeline again.
    return state(resource).nextIssueCycle;
}

void
ResourceTracker::issue(Resource resource, Cycles now)
{
    // Reserve one issue slot and record that another operation is in flight.
    ResourceState &resource_state = mutableState(resource);
    panic_if(!canIssue(resource, now),
             "AMX %s resource issued before it became available",
             resourceName(resource));

    // Throughput controls the next issue time; it does not wait for completion.
    resource_state.nextIssueCycle = now + resource_state.initiationInterval;
    ++resource_state.inFlight;
    ++resource_state.issueCount;
}

void
ResourceTracker::complete(Resource resource)
{
    // Record that one issued operation is no longer in flight.
    ResourceState &resource_state = mutableState(resource);
    panic_if(resource_state.inFlight == 0,
             "AMX %s resource completed with no instruction in flight",
             resourceName(resource));
    --resource_state.inFlight;
}

const ResourceState &
ResourceTracker::state(Resource resource) const
{
    // Provide read-only access to one pipeline's timing and counters.
    return states[resourceIndex(resource)];
}

ResourceState &
ResourceTracker::mutableState(Resource resource)
{
    // Provide internal write access to one pipeline's timing and counters.
    return states[resourceIndex(resource)];
}

const char *
resourceName(Resource resource)
{
    // Provide readable names for debug and error messages.
    switch (resource) {
        case Resource::TileLoad:
            return "tile-load";
        case Resource::DotProduct:
            return "dot-product";
        case Resource::TileZero:
            return "tile-zero";
        case Resource::TileStore:
            return "tile-store";
        case Resource::Count:
            break;
    }

    panic("AMX has no name for an invalid resource");
}

} // namespace amx
} // namespace gem5
