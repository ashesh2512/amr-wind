#ifndef StokesOscillatingSphere_H
#define StokesOscillatingSphere_H

#include "amr-wind/core/Physics.H"
#include "amr-wind/core/Field.H"
#include "amr-wind/CFDSim.H"
#include "amr-wind/utilities/trig_ops.H"

#include <AMReX_GpuComplex.H>

namespace amr_wind {
namespace sop {

namespace {
struct UExact
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real operator()(
        const amrex::GpuComplex<amrex::Real>,
        const amrex::GpuComplex<amrex::Real>,
        const amrex::GpuComplex<amrex::Real>,
        const amrex::GpuComplex<amrex::Real>,
        const amrex::Vector<amrex::Real>,
        const amrex::Real,
        const amrex::Real,
        const amrex::Real,
        const amrex::Real,
        const amrex::Real) const;
    const int m_comp{0};
};

struct VExact
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real operator()(
        const amrex::GpuComplex<amrex::Real>,
        const amrex::GpuComplex<amrex::Real>,
        const amrex::GpuComplex<amrex::Real>,
        const amrex::GpuComplex<amrex::Real>,
        const amrex::Vector<amrex::Real>,
        const amrex::Real,
        const amrex::Real,
        const amrex::Real,
        const amrex::Real,
        const amrex::Real) const;
    const int m_comp{1};
};

struct WExact
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real operator()(
        const amrex::GpuComplex<amrex::Real>,
        const amrex::GpuComplex<amrex::Real>,
        const amrex::GpuComplex<amrex::Real>,
        const amrex::GpuComplex<amrex::Real>,
        const amrex::Vector<amrex::Real>,
        const amrex::Real,
        const amrex::Real,
        const amrex::Real,
        const amrex::Real,
        const amrex::Real) const;
    const int m_comp{2};
};

struct GpxExact
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real operator()(
        const amrex::GpuComplex<amrex::Real>,
        const amrex::GpuComplex<amrex::Real>,
        const amrex::GpuComplex<amrex::Real>,
        const amrex::GpuComplex<amrex::Real>,
        const amrex::Vector<amrex::Real>,
        const amrex::Real,
        const amrex::Real,
        const amrex::Real,
        const amrex::Real,
        const amrex::Real) const;
    const int m_comp{0};
};

struct GpyExact
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real operator()(
        const amrex::GpuComplex<amrex::Real>,
        const amrex::GpuComplex<amrex::Real>,
        const amrex::GpuComplex<amrex::Real>,
        const amrex::GpuComplex<amrex::Real>,
        const amrex::Vector<amrex::Real>,
        const amrex::Real,
        const amrex::Real,
        const amrex::Real,
        const amrex::Real,
        const amrex::Real) const;
    const int m_comp{1};
};

struct GpzExact
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real operator()(
        const amrex::GpuComplex<amrex::Real>,
        const amrex::GpuComplex<amrex::Real>,
        const amrex::GpuComplex<amrex::Real>,
        const amrex::GpuComplex<amrex::Real>,
        const amrex::Vector<amrex::Real>,
        const amrex::Real,
        const amrex::Real,
        const amrex::Real,
        const amrex::Real,
        const amrex::Real) const;
    const int m_comp{2};
};

} // namespace

/** Stokes oscillating sphere physics
 *  \ingroup physics
 */
class StokesOscillatingSphere : public Physics::Register<StokesOscillatingSphere>
{
public:
    static const std::string identifier() { return "StokesOscillatingSphere"; }

    StokesOscillatingSphere(const CFDSim& sim);

    virtual ~StokesOscillatingSphere() = default;

    void initialize_fields(int level, const amrex::Geometry& geom) override;

    template <typename T>
    amrex::Real compute_error(const Field&);

    void post_init_actions() override;

    void post_regrid_actions() override {}

    void pre_advance_work() override {}

    void post_advance_work() override;

private:
    const amr_wind::SimTime& m_time;
    const amr_wind::CFDSim& m_sim;
    const FieldRepo& m_repo;
    const amrex::AmrCore& m_mesh;
    Field& m_velocity;
    Field& m_gradp;
    Field& m_density;
    void output_error();

    //! Oscillating frequency
    const amrex::Real m_eps{1.e-8};

    //! Oscillating frequency
    const amrex::Real m_omega{2*utils::pi()};

    //! Default viscosity value
    amrex::Real m_nu{1.0};

    //! Radius of sphere
    amrex::Real m_r0{0.5};

    //! Initial density value
    amrex::Real m_rho{1.0};

    //! Center of sphere
    amrex::Vector<amrex::Real> m_cen{0.0,0.0,0.0};

    //! Default velocity coefficient vector
    amrex::Vector<amrex::Real> m_v_coeff{1.0,0.0,0.0};

    //! Analytical solution related complex constants
    amrex::GpuComplex<amrex::Real> m_b0{0.0,0.0};
    amrex::GpuComplex<amrex::Real> m_lambda{0.0,0.0};
    amrex::GpuComplex<amrex::Real> m_q0{0.0,0.0};

    //! output precision
    const int m_w{18};

    //! error log file
    std::string m_output_fname{"sop.log"};
};
} // namespace sop
} // namespace amr_wind

#endif /* StokesOscillatingSphere_H */
