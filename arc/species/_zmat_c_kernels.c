/*
 * _zmat_c_kernels.c — fast geometry kernels for arc.species.zmat
 *
 * All functions take individual x,y,z doubles per atom.
 * This lets Python callers use tuple-unpacking (*coords[i]) instead of
 * building numpy arrays, eliminating the dominant overhead in the hot path.
 *
 * Compiled by `make compile` with:
 *   cc -O3 -fno-math-errno -fno-trapping-math -shared -fPIC \
 *      -o _zmat_c_kernels.so _zmat_c_kernels.c -lm
 *
 * Deliberately built for the portable baseline ISA (no -march=native) and without
 * -ffast-math, so that NaN results stay observable -- callers rely on a NaN dihedral
 * to detect degenerate/collinear geometry.
 */

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── helpers ──────────────────────────────────────────────────────────────── */

static inline double _dot3(double ax, double ay, double az,
                            double bx, double by, double bz) {
    return ax*bx + ay*by + az*bz;
}

static inline double _len3(double x, double y, double z) {
    return sqrt(x*x + y*y + z*z);
}

/* ── bond angle (degrees), full double precision ──────────────────────────── */
/*
 * Mirrors vectors.calculate_angle:
 *   v1 = B - A,  v2 = B - C   (centre atom is B = second argument)
 *
 * Unlike the two kernels below this one does NOT truncate to float32: its numpy
 * counterpart was switched to float64 so that both paths agree exactly, and so
 * that the angles ARC writes into z-matrices are not carrying ~8e-4 deg of
 * float32 noise against a 1e-3 deg consolidation tolerance.
 */
double zmat_a(double ax, double ay, double az,
              double bx, double by, double bz,
              double cx, double cy, double cz) {
    double v1x = bx - ax, v1y = by - ay, v1z = bz - az;
    double v2x = bx - cx, v2y = by - cy, v2z = bz - cz;
    double len1 = _len3(v1x, v1y, v1z);
    double len2 = _len3(v2x, v2y, v2z);
    if (len1 == 0. || len2 == 0.) return 0. / 0.;   /* NaN: get_angle divides by zero */
    double cosine = _dot3(v1x/len1, v1y/len1, v1z/len1, v2x/len2, v2y/len2, v2z/len2);
    if (cosine >  1.) cosine =  1.;
    if (cosine < -1.) cosine = -1.;
    return acos(cosine) * (180. / M_PI);
}

/* ── float32-input kernels (matching np.asarray(coords, float32)) ──────────── */
/*
 * calculate_distance and calculate_dihedral_angle still truncate their
 * coordinates to float32 before computing, as they have since dd452891 (2020).
 * These mirror that exactly so the C and numpy paths cannot disagree. Lifting
 * them to float64 the way calculate_angle was is a worthwhile follow-up, but it
 * moves every distance and dihedral ARC writes, so it wants its own review.
 */

double zmat_r_f32(double ax, double ay, double az,
                  double bx, double by, double bz) {
    float fax=(float)ax, fay=(float)ay, faz=(float)az;
    float fbx=(float)bx, fby=(float)by, fbz=(float)bz;
    float dx=fbx-fax, dy=fby-fay, dz=fbz-faz;
    return (double)sqrtf(dx*dx + dy*dy + dz*dz);
}

double zmat_d_f32(double ax, double ay, double az,
                  double bx, double by, double bz,
                  double cx, double cy, double cz,
                  double dx, double dy, double dz) {
    /* Compute difference vectors in float32 (matches np.asarray(coords, float32)).
     * Then convert to double for cross products, matching get_dihedral() which
     * immediately promotes its float32 input to np.float64 before any arithmetic. */
    float fax=(float)ax, fay=(float)ay, faz=(float)az;
    float fbx=(float)bx, fby=(float)by, fbz=(float)bz;
    float fcx=(float)cx, fcy=(float)cy, fcz=(float)cz;
    float fdx=(float)dx, fdy=(float)dy, fdz=(float)dz;
    double v1x=(double)(fbx-fax), v1y=(double)(fby-fay), v1z=(double)(fbz-faz);
    double v2x=(double)(fcx-fbx), v2y=(double)(fcy-fby), v2z=(double)(fcz-fbz);
    double v3x=(double)(fdx-fcx), v3y=(double)(fdy-fcy), v3z=(double)(fdz-fcz);
    /* n1 = v2 × v1 */
    double n1x=v2y*v1z-v2z*v1y, n1y=v2z*v1x-v2x*v1z, n1z=v2x*v1y-v2y*v1x;
    double nm1=_len3(n1x,n1y,n1z);
    if (nm1<1e-8) return 0./0.;
    n1x/=nm1; n1y/=nm1; n1z/=nm1;
    /* n2 = v3 × v2 */
    double n2x=v3y*v2z-v3z*v2y, n2y=v3z*v2x-v3x*v2z, n2z=v3x*v2y-v3y*v2x;
    double nm2=_len3(n2x,n2y,n2z);
    if (nm2<1e-8) return 0./0.;
    n2x/=nm2; n2y/=nm2; n2z/=nm2;
    double cosine=_dot3(n1x,n1y,n1z,n2x,n2y,n2z);
    if (cosine> 1.) cosine= 1.;
    if (cosine<-1.) cosine=-1.;
    double dihedral=acos(cosine);
    if (_dot3(n1x,n1y,n1z,v3x,v3y,v3z) > 0.)
        dihedral=2.*M_PI-dihedral;
    return dihedral*(180./M_PI);
}

