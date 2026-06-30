/*
 * _zmat_c_kernels.c — fast geometry kernels for arc.species.zmat
 *
 * All functions take individual x,y,z doubles per atom.
 * This lets Python callers use tuple-unpacking (*coords[i]) instead of
 * building numpy arrays, eliminating the dominant overhead in the hot path.
 *
 * Compiled with:
 *   gcc -O3 -march=native -ffast-math -shared -fPIC -o _zmat_c_kernels.so \
 *       _zmat_c_kernels.c -lm
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

/* ── distance ─────────────────────────────────────────────────────────────── */

double zmat_r(double ax, double ay, double az,
              double bx, double by, double bz) {
    double dx = bx - ax, dy = by - ay, dz = bz - az;
    return _len3(dx, dy, dz);
}

/* ── bond angle (degrees) ─────────────────────────────────────────────────── */
/*
 * Mirrors calculate_angle in vectors.py:
 *   v1 = B - A,  v2 = B - C   (centre atom is B = second argument)
 */
double zmat_a(double ax, double ay, double az,
              double bx, double by, double bz,
              double cx, double cy, double cz) {
    double v1x = bx - ax, v1y = by - ay, v1z = bz - az;
    double v2x = bx - cx, v2y = by - cy, v2z = bz - cz;
    double len1 = _len3(v1x, v1y, v1z);
    double len2 = _len3(v2x, v2y, v2z);
    if (len1 == 0. || len2 == 0.) return 0.;
    double cosine = _dot3(v1x, v1y, v1z, v2x, v2y, v2z) / (len1 * len2);
    if (cosine >  1.) cosine =  1.;
    if (cosine < -1.) cosine = -1.;
    return acos(cosine) * (180. / M_PI);
}

/* ── dihedral angle (degrees) ─────────────────────────────────────────────── */
/*
 * Mirrors calculate_dihedral_angle → get_dihedral in vectors.py:
 *   v1 = B - A,  v2 = C - B,  v3 = D - C   (four atoms A-B-C-D)
 */
double zmat_d(double ax, double ay, double az,
              double bx, double by, double bz,
              double cx, double cy, double cz,
              double dx, double dy, double dz) {
    double v1x = bx-ax, v1y = by-ay, v1z = bz-az;
    double v2x = cx-bx, v2y = cy-by, v2z = cz-bz;
    double v3x = dx-cx, v3y = dy-cy, v3z = dz-cz;

    /* n1 = v2 × v1 */
    double n1x = v2y*v1z - v2z*v1y;
    double n1y = v2z*v1x - v2x*v1z;
    double n1z = v2x*v1y - v2y*v1x;
    double nm1 = _len3(n1x, n1y, n1z);
    if (nm1 < 1e-8) return 0. / 0.;   /* NaN for colinear */
    n1x /= nm1;  n1y /= nm1;  n1z /= nm1;

    /* n2 = v3 × v2 */
    double n2x = v3y*v2z - v3z*v2y;
    double n2y = v3z*v2x - v3x*v2z;
    double n2z = v3x*v2y - v3y*v2x;
    double nm2 = _len3(n2x, n2y, n2z);
    if (nm2 < 1e-8) return 0. / 0.;
    n2x /= nm2;  n2y /= nm2;  n2z /= nm2;

    double cosine = _dot3(n1x, n1y, n1z, n2x, n2y, n2z);
    if (cosine >  1.) cosine =  1.;
    if (cosine < -1.) cosine = -1.;
    double dihedral = acos(cosine);

    if (_dot3(n1x, n1y, n1z, v3x, v3y, v3z) > 0.)
        dihedral = 2. * M_PI - dihedral;

    return dihedral * (180. / M_PI);
}

/* ── float32-precision variants (match original np.asarray(coords, float32)) ─ */
/*
 * The original vectors.calculate_param converts coordinates to float32 before
 * computing, which aligns with the z-matrix consolidation tolerance (1e-4 Å).
 * These variants truncate inputs to float32 first so that consolidation
 * produces the same grouped parameters as the original Python code.
 */

double zmat_r_f32(double ax, double ay, double az,
                  double bx, double by, double bz) {
    float fax=(float)ax, fay=(float)ay, faz=(float)az;
    float fbx=(float)bx, fby=(float)by, fbz=(float)bz;
    float dx=fbx-fax, dy=fby-fay, dz=fbz-faz;
    return (double)sqrtf(dx*dx + dy*dy + dz*dz);
}

double zmat_a_f32(double ax, double ay, double az,
                  double bx, double by, double bz,
                  double cx, double cy, double cz) {
    float fax=(float)ax, fay=(float)ay, faz=(float)az;
    float fbx=(float)bx, fby=(float)by, fbz=(float)bz;
    float fcx=(float)cx, fcy=(float)cy, fcz=(float)cz;
    float v1x=fbx-fax, v1y=fby-fay, v1z=fbz-faz;
    float v2x=fbx-fcx, v2y=fby-fcy, v2z=fbz-fcz;
    /* Sum of squares in float32 (matches Python sum([v*v for v in v])), then
     * sqrt in double (matches math.sqrt → get_vector_length in vectors.py) */
    float sq1 = v1x*v1x + v1y*v1y + v1z*v1z;
    float sq2 = v2x*v2x + v2y*v2y + v2z*v2z;
    double len1 = sqrt((double)sq1), len2 = sqrt((double)sq2);
    if (len1 == 0. || len2 == 0.) return 0.;
    /* Divide: float32(vi / double_len) — matches numpy float32/Python_float → float32 */
    float u1x=(float)(v1x/len1), u1y=(float)(v1y/len1), u1z=(float)(v1z/len1);
    float u2x=(float)(v2x/len2), u2y=(float)(v2y/len2), u2z=(float)(v2z/len2);
    /* Float32 dot product (matches np.dot(float32_array, float32_array)) */
    float cosine = u1x*u2x + u1y*u2y + u1z*u2z;
    if (cosine > 1.f) cosine = 1.f;
    if (cosine <-1.f) cosine =-1.f;
    /* arccos in float32, degree conversion in float32 (matches np.arccos(float32)*float) */
    return (double)(acosf(cosine) * (180.0f / (float)M_PI));
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

/* ── SN-NeRF atom placement ───────────────────────────────────────────────── */
/*
 * Given atoms A, B, C already placed, add atom D with:
 *   bond length C-D  = cd_len
 *   angle B-C-D      = bcd_deg  (degrees)
 *   dihedral A-B-C-D = abcd_deg (degrees)
 *
 * Implements the SN-NeRF algorithm (Parsons et al. 2005, J. Comput. Chem.).
 * Mirrors _add_nth_atom_to_coords in zmat.py for i >= 3.
 */
void zmat_nerf(double ax, double ay, double az,
               double bx, double by, double bz,
               double cx, double cy, double cz,
               double cd_len, double bcd_deg, double abcd_deg,
               double *rdx, double *rdy, double *rdz) {
    /* B→C vector, normalised */
    double bcx = cx - bx, bcy = cy - by, bcz = cz - bz;
    double bc_len = _len3(bcx, bcy, bcz);
    double ubcx = bcx / bc_len, ubcy = bcy / bc_len, ubcz = bcz / bc_len;

    /* A→B vector (not normalised — only direction needed for n) */
    double abx = bx - ax, aby = by - ay, abz = bz - az;

    /* n = ab × ubc  (normal to the A-B-C plane) */
    double nx = aby*ubcz - abz*ubcy;
    double ny = abz*ubcx - abx*ubcz;
    double nz = abx*ubcy - aby*ubcx;
    double nlen = _len3(nx, ny, nz);
    double unx = nx / nlen, uny = ny / nlen, unz = nz / nlen;

    /* un × ubc  (third column of rotation matrix) */
    double ucx = uny*ubcz - unz*ubcy;
    double ucy = unz*ubcx - unx*ubcz;
    double ucz = unx*ubcy - uny*ubcx;

    /* D in the local frame: [-cd·cos(bcd), cd·sin(bcd)·cos(abcd), cd·sin(bcd)·sin(abcd)] */
    double bcd  = bcd_deg  * (M_PI / 180.);
    double abcd = abcd_deg * (M_PI / 180.);
    double sin_bcd = sin(bcd);
    double d0 = -cd_len * cos(bcd);
    double d1 =  cd_len * sin_bcd * cos(abcd);
    double d2 =  cd_len * sin_bcd * sin(abcd);

    /* Rotate into the A-B-C frame and translate to C */
    *rdx = ubcx*d0 + ucx*d1 + unx*d2 + cx;
    *rdy = ubcy*d0 + ucy*d1 + uny*d2 + cy;
    *rdz = ubcz*d0 + ucz*d1 + unz*d2 + cz;
}
