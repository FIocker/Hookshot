// (hookshot) gpu/vk/framegen_methods.h: the Frame generation method row's stepping and what vk_framegen.cpp tries.
#include <cstdio>
#include "../gpu/vk/framegen_methods.h"

namespace fgm = mkwvk::framegen_methods;
static int failures;
#define CHECK(name, c) do { if (!(c)) { std::printf("FAIL: %s\n", name); ++failures; } } while (0)

int main() {
    using namespace fgm;
    const unsigned all = bit(AUTO) | bit(DLSS) | bit(FSR4) | bit(FSR3) | bit(BUILTIN);
    const unsigned fsr3 = bit(FSR3);   // (Auto and Built-in are always there)
    CHECK("steps through all five", step(AUTO, 1, all) == DLSS && step(DLSS, 1, all) == FSR4 && step(FSR4, 1, all) == FSR3 &&
                                        step(FSR3, 1, all) == BUILTIN && step(BUILTIN, 1, all) == AUTO);
    CHECK("steps back", step(AUTO, -1, all) == BUILTIN && step(DLSS, -1, all) == AUTO);
    CHECK("skips what does not run", step(AUTO, 1, fsr3) == FSR3 && step(FSR3, 1, fsr3) == BUILTIN &&
                                         step(BUILTIN, 1, fsr3) == AUTO && step(AUTO, -1, fsr3) == BUILTIN);
    CHECK("nothing but the always ones", step(AUTO, 1, 0) == BUILTIN && step(BUILTIN, 1, 0) == AUTO);
    CHECK("a bad value steps from Auto", step(9, 1, all) == DLSS && step(-3, -1, all) == BUILTIN);
    CHECK("a choice that runs is kept", effective(FSR3, fsr3) == FSR3 && effective(BUILTIN, 0) == BUILTIN);
    CHECK("a choice that does not run is Auto", effective(DLSS, fsr3) == AUTO && effective(FSR4, 0) == AUTO &&
                                                    effective(7, all) == AUTO);
    CHECK("Auto tries every vendor method", tries(AUTO, DLSS) && tries(AUTO, FSR4) && tries(AUTO, FSR3));
    CHECK("a chosen method alone", tries(FSR3, FSR3) && !tries(FSR3, DLSS) && !tries(FSR3, FSR4));
    CHECK("built-in tries no vendor method", !tries(BUILTIN, DLSS) && !tries(BUILTIN, FSR4) && !tries(BUILTIN, FSR3));
    // FSR 3.1 makes the midpoint only: 2x sets, never 3x and above (the same picture at two times would judder)
    CHECK("FSR 3.1 makes 2x sets only", makes_set(FSR3, 2) && !makes_set(FSR3, 3) && !makes_set(FSR3, 8));
    CHECK("the others make every set", makes_set(DLSS, 4) && makes_set(FSR4, 5) && makes_set(BUILTIN, 8));
    CHECK("Auto tries the best first", first(AUTO, all, 2) == DLSS && first(AUTO, bit(FSR4) | bit(FSR3), 3) == FSR4 &&
                                           first(AUTO, fsr3, 2) == FSR3);
    CHECK("FSR 3.1 at 3x leaves the set to the built-in blend", first(AUTO, fsr3, 3) == BUILTIN &&
                                                                  first(FSR3, all, 4) == BUILTIN && first(FSR3, all, 2) == FSR3);
    CHECK("built-in and what does not run", first(BUILTIN, all, 2) == BUILTIN && first(DLSS, fsr3, 2) == BUILTIN);
    if (failures) return 1;
    std::printf("PASS: the Frame generation method row steps over what runs, and Auto or one method is tried\n");
    return 0;
}
