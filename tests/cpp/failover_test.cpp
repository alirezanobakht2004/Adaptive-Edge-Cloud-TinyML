#include <cassert>
#include <cstdlib>
#include "policy/failover.h"
int main(int argc, char** argv) {
    assert(argc == 2);
    const int scenario = std::atoi(argv[1]);
    policy::FailoverState state;
    if (scenario == 0) {
        assert(state.begin(0, false, false, 2, 1));
        assert(state.complete && !state.failover && state.finalClass == 2);
    } else if (scenario == 1) {
        assert(state.begin(1, true, true, 1, 1)); state.cloudSuccess(2);
        assert(state.complete && !state.failover && state.effectiveAction == 1 && state.finalClass == 2);
    } else if (scenario == 2 || scenario == 3) {
        assert(state.begin(1, scenario == 3, false, 1, 1));
        assert(state.complete && state.failover && state.finalClass == 1);
        assert(state.reason == (scenario == 2 ? policy::FailoverReason::WIFI_UNAVAILABLE : policy::FailoverReason::MQTT_UNAVAILABLE));
    } else if (scenario == 4 || scenario == 5) {
        assert(state.begin(1, true, true, 1, 1));
        state.fallback(scenario == 4 ? policy::FailoverReason::CLOUD_RESPONSE_TIMEOUT : policy::FailoverReason::CLOUD_PUBLISH_FAILED);
        state.cloudSuccess(4); // A late response must not replace a completed fallback.
        assert(state.complete && state.failover && state.finalClass == 1 && state.effectiveAction == 0);
    } else if (scenario == 6) {
        assert(state.begin(1, true, true, 1, 1)); state.fallback(policy::FailoverReason::CLOUD_RESPONSE_TIMEOUT);
        assert(state.begin(1, true, true, 3, 1)); state.cloudSuccess(4);
        assert(state.complete && !state.failover && state.finalClass == 4);
    } else if (scenario == 7) {
        assert(!state.begin(1, true, true, 1, 2)); return 0;
    } else {
        assert(!policy::requestExpired(100, 0xfffffff0U, 3000));
        assert(policy::requestExpired(3000000U, 0xfffffff0U, 3000)); return 0;
    }
    assert(state.localInferenceCount == 1);
}
