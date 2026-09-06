#ifndef MOCAP_VISIBILITY_HPP
#define MOCAP_VISIBILITY_HPP

// everything in the library is hidden by default, so the public entry points
// have to opt back in
#define MOCAP_API __attribute__((visibility("default")))

#endif // MOCAP_VISIBILITY_HPP
