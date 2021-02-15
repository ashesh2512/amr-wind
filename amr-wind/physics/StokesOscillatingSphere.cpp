#include "amr-wind/physics/StokesOscillatingSphere.H"
#include "amr-wind/CFDSim.H"
#include "AMReX_iMultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParmParse.H"

namespace amr_wind {
namespace sop {

namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real UExact::operator()(
    const amrex::GpuComplex<amrex::Real> lambda,
    const amrex::GpuComplex<amrex::Real> b_coeff,
    const amrex::GpuComplex<amrex::Real> q_coeff,
    const amrex::GpuComplex<amrex::Real>,
    const amrex::Vector<amrex::Real> v_coeff,
    const amrex::Real r0,
    const amrex::Real x,
    const amrex::Real y,
    const amrex::Real z,
    const amrex::Real eps) const
{
    // get distance of point w.r.t. sphere center
    amrex::Real dist = std::sqrt(std::pow(x,2)+std::pow(y,2)+std::pow(z,2));

    // solution does not exist inside sphere
    if(dist < (r0-eps)) {
      return 0.0;
    }

    // dimensionless distance metrics
    amrex::GpuComplex<amrex::Real> cap_r  = lambda*dist/r0;
    amrex::GpuComplex<amrex::Real> cap_r2 = amrex::pow(cap_r,2.);
    amrex::GpuComplex<amrex::Real> exp_r  = amrex::exp(-cap_r);

    amrex::Real v_dot_x = v_coeff[0]*x+v_coeff[1]*y+v_coeff[2]*z;

    // different components of 1st term
    amrex::GpuComplex<amrex::Real> b1 = (2.*exp_r*(1.+1./cap_r+1./cap_r2) - 2./cap_r2) * v_coeff[0] / (dist+eps);
    amrex::GpuComplex<amrex::Real> b2 = (6./cap_r2 - 2.*exp_r*(1.+3./cap_r+3./cap_r2)) * v_dot_x * x / (std::pow(dist,3.)+eps);
    amrex::GpuComplex<amrex::Real> t1 = b_coeff * (b1 + b2);

    // different components of 2nd term
    amrex::GpuComplex<amrex::Real> q1 = -exp_r * (1.+cap_r+cap_r2) * v_coeff[0] / (std::pow(dist,3.)+eps);
    amrex::GpuComplex<amrex::Real> q2 = 3. * exp_r * (1.+cap_r+cap_r2/3.) * v_dot_x * x / (std::pow(dist,5.)+eps);
    amrex::GpuComplex<amrex::Real> t2 = q_coeff * (q1 + q2);

    // extract real parts
    return (t1.real() + t2.real());
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real VExact::operator()(
    const amrex::GpuComplex<amrex::Real> lambda,
    const amrex::GpuComplex<amrex::Real> b_coeff,
    const amrex::GpuComplex<amrex::Real> q_coeff,
    const amrex::GpuComplex<amrex::Real>,
    const amrex::Vector<amrex::Real> v_coeff,
    const amrex::Real r0,
    const amrex::Real x,
    const amrex::Real y,
    const amrex::Real z,
    const amrex::Real eps) const
{
    // get distance of point w.r.t. sphere center
    amrex::Real dist = std::sqrt(std::pow(x,2)+std::pow(y,2)+std::pow(z,2));

    // solution does not exist inside sphere
    if(dist < (r0-eps)) {
      return 0.0;
    }

    // dimensionless distance metrics
    amrex::GpuComplex<amrex::Real> cap_r  = lambda*dist/r0;
    amrex::GpuComplex<amrex::Real> cap_r2 = amrex::pow(cap_r,2.);
    amrex::GpuComplex<amrex::Real> exp_r  = amrex::exp(-cap_r);

    amrex::Real v_dot_x = v_coeff[0]*x+v_coeff[1]*y+v_coeff[2]*z;

    // different components of 1st term
    amrex::GpuComplex<amrex::Real> b1 = (2.*exp_r*(1.+1./cap_r+1./cap_r2) - 2./cap_r2) * v_coeff[1] / (dist+eps);
    amrex::GpuComplex<amrex::Real> b2 = (6./cap_r2 - 2.*exp_r*(1.+3./cap_r+3./cap_r2)) * v_dot_x * y / (std::pow(dist,3.)+eps);
    amrex::GpuComplex<amrex::Real> t1 = b_coeff * (b1 + b2);

    // different components of 2nd term
    amrex::GpuComplex<amrex::Real> q1 = -exp_r * (1.+cap_r+cap_r2) * v_coeff[1] / (std::pow(dist,3.)+eps);
    amrex::GpuComplex<amrex::Real> q2 = 3. * exp_r * (1.+cap_r+cap_r2/3.) * v_dot_x * y / (std::pow(dist,5.)+eps);
    amrex::GpuComplex<amrex::Real> t2 = q_coeff * (q1 + q2);

    // extract real parts
    return (t1.real() + t2.real());
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real WExact::operator()(
    const amrex::GpuComplex<amrex::Real> lambda,
    const amrex::GpuComplex<amrex::Real> b_coeff,
    const amrex::GpuComplex<amrex::Real> q_coeff,
    const amrex::GpuComplex<amrex::Real>,
    const amrex::Vector<amrex::Real> v_coeff,
    const amrex::Real r0,
    const amrex::Real x,
    const amrex::Real y,
    const amrex::Real z,
    const amrex::Real eps) const
{
    // get distance of point w.r.t. sphere center
    amrex::Real dist = std::sqrt(std::pow(x,2)+std::pow(y,2)+std::pow(z,2));

    // solution does not exist inside sphere
    if(dist < (r0-eps)) {
      return 0.0;
    }

    // dimensionless distance metrics
    amrex::GpuComplex<amrex::Real> cap_r  = lambda*dist/r0;
    amrex::GpuComplex<amrex::Real> cap_r2 = amrex::pow(cap_r,2.);
    amrex::GpuComplex<amrex::Real> exp_r  = amrex::exp(-cap_r);

    amrex::Real v_dot_x = v_coeff[0]*x+v_coeff[1]*y+v_coeff[2]*z;

    // different components of 1st term
    amrex::GpuComplex<amrex::Real> b1 = (2.*exp_r*(1.+1./cap_r+1./cap_r2) - 2./cap_r2) * v_coeff[2] / (dist+eps);
    amrex::GpuComplex<amrex::Real> b2 = (6./cap_r2 - 2.*exp_r*(1.+3./cap_r+3./cap_r2)) * v_dot_x * z / (std::pow(dist,3.)+eps);
    amrex::GpuComplex<amrex::Real> t1 = b_coeff * (b1 + b2);

    // different components of 2nd term
    amrex::GpuComplex<amrex::Real> q1 = -exp_r * (1.+cap_r+cap_r2) * v_coeff[2] / (std::pow(dist,3.)+eps);
    amrex::GpuComplex<amrex::Real> q2 = 3. * exp_r * (1.+cap_r+cap_r2/3.) * v_dot_x * z / (std::pow(dist,5.)+eps);
    amrex::GpuComplex<amrex::Real> t2 = q_coeff * (q1 + q2);

    // extract real parts
    return (t1.real() + t2.real());
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real GpxExact::operator()(
    const amrex::GpuComplex<amrex::Real>,
    const amrex::GpuComplex<amrex::Real>,
    const amrex::GpuComplex<amrex::Real>,
    const amrex::GpuComplex<amrex::Real> p_coeff,
    const amrex::Vector<amrex::Real> v_coeff,
    const amrex::Real r0,
    const amrex::Real x,
    const amrex::Real y,
    const amrex::Real z,
    const amrex::Real eps) const
{
    // get distance of point w.r.t. sphere center
    amrex::Real dist = std::sqrt(std::pow(x,2)+std::pow(y,2)+std::pow(z,2));

    // solution does not exist inside sphere
    if(dist < (r0-eps)) {
      return 0.0;
    }

    amrex::Real v_dot_x = v_coeff[0]*x+v_coeff[1]*y+v_coeff[2]*z;

    return (p_coeff*(v_coeff[0]/(std::pow(dist,3.)+eps) - 3.*v_dot_x*x/(std::pow(dist,5.)+eps))).real();
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real GpyExact::operator()(
    const amrex::GpuComplex<amrex::Real>,
    const amrex::GpuComplex<amrex::Real>,
    const amrex::GpuComplex<amrex::Real>,
    const amrex::GpuComplex<amrex::Real> p_coeff,
    const amrex::Vector<amrex::Real> v_coeff,
    const amrex::Real r0,
    const amrex::Real x,
    const amrex::Real y,
    const amrex::Real z,
    const amrex::Real eps) const
{
    // get distance of point w.r.t. sphere center
    amrex::Real dist = std::sqrt(std::pow(x,2)+std::pow(y,2)+std::pow(z,2));

    // solution does not exist inside sphere
    if(dist < (r0-eps)) {
      return 0.0;
    }

    amrex::Real v_dot_x = v_coeff[0]*x+v_coeff[1]*y+v_coeff[2]*z;

    return (p_coeff*(v_coeff[1]/(std::pow(dist,3.)+eps) - 3.*v_dot_x*y/(std::pow(dist,5.)+eps))).real();
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real GpzExact::operator()(
    const amrex::GpuComplex<amrex::Real>,
    const amrex::GpuComplex<amrex::Real>,
    const amrex::GpuComplex<amrex::Real>,
    const amrex::GpuComplex<amrex::Real> p_coeff,
    const amrex::Vector<amrex::Real> v_coeff,
    const amrex::Real r0,
    const amrex::Real x,
    const amrex::Real y,
    const amrex::Real z,
    const amrex::Real eps) const
{
    // get distance of point w.r.t. sphere center
    amrex::Real dist = std::sqrt(std::pow(x,2)+std::pow(y,2)+std::pow(z,2));

    // solution does not exist inside sphere
    if(dist < (r0-eps)) {
      return 0.0;
    }

    amrex::Real v_dot_x = v_coeff[0]*x+v_coeff[1]*y+v_coeff[2]*z;

    return (p_coeff*(v_coeff[2]/(std::pow(dist,3.)+eps) - 3.*v_dot_x*z/(std::pow(dist,5.)+eps))).real();
}

} // namespace

StokesOscillatingSphere::StokesOscillatingSphere(const CFDSim& sim)
    : m_time(sim.time())
    , m_sim(sim)
    , m_repo(sim.repo())
    , m_mesh(sim.mesh())
    , m_velocity(sim.repo().get_field("velocity"))
    , m_gradp(sim.repo().get_field("gp"))
    , m_density(sim.repo().get_field("density"))
{
    amrex::ParmParse pp("SOP");
    pp.query("density", m_rho);
    pp.queryarr("sphere_center", m_cen, 0, AMREX_SPACEDIM);
    pp.query("sphere_radius", m_r0);
    pp.queryarr("vel_coeff", m_v_coeff, 0, AMREX_SPACEDIM);
    pp.query("error_log_file", m_output_fname);
    {
        amrex::ParmParse pp("transport");
        pp.query("viscosity", m_nu);
    }

    // construct constants related to the analytical solution
    m_lambda = amrex::GpuComplex<amrex::Real>{1.,-1.}*m_r0*std::sqrt(0.5*std::abs(m_omega)/m_nu);
    m_b0     = 6.*utils::pi()*m_nu*m_r0*(1.+m_lambda+amrex::pow(m_lambda,2.)/3.);
    m_q0     = -6.*utils::pi()*std::pow(m_r0,3.) *
               (amrex::exp(m_lambda)-1.-m_lambda-amrex::pow(m_lambda,2.)/3.)/m_lambda/m_lambda;

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream f;
        f.open(m_output_fname.c_str());
        f << std::setw(m_w) << "time" << std::setw(m_w) << "L2_u"
          << std::setw(m_w) << "L2_v" << std::setw(m_w) << "L2_w"
          << std::setw(m_w) << "L2_gpx" << std::setw(m_w) << "L2_gpy"
          << std::setw(m_w) << "L2_gpz" << std::endl;
        f.close();
    }
}

/** Initialize the velocity and density fields at the beginning of the
 *  simulation.
 */
void StokesOscillatingSphere::initialize_fields(
    int level, const amrex::Geometry& geom)
{
    const auto eps     = m_eps;
    const auto lambda  = m_lambda;
    const auto r0      = m_r0;
    const auto time    = m_time.new_time();
    const auto v_coeff = m_v_coeff;

    // evaluate translated center of sphere
    const auto cen_x = m_cen[0] - (amrex::exp(
                                  amrex::GpuComplex<amrex::Real>{0.,-1.}*m_omega*time)/m_omega*v_coeff[0]).real();
    const auto cen_y = m_cen[1] - (amrex::exp(
                                  amrex::GpuComplex<amrex::Real>{0.,-1.}*m_omega*time)/m_omega*v_coeff[1]).real();
    const auto cen_z = m_cen[2] - (amrex::exp(
                                  amrex::GpuComplex<amrex::Real>{0.,-1.}*m_omega*time)/m_omega*v_coeff[2]).real();

    // velocity and pressure coefficients
    amrex::GpuComplex<amrex::Real> b_coeff = amrex::GpuComplex<amrex::Real>{0.,1.} *
                                             amrex::exp(amrex::GpuComplex<amrex::Real>{0.,-1.}*m_omega*0.0) *
                                             m_b0/(8.*utils::pi()*m_nu);
    amrex::GpuComplex<amrex::Real> q_coeff = amrex::GpuComplex<amrex::Real>{0.,1.} *
                                             amrex::exp(amrex::GpuComplex<amrex::Real>{0.,-1.}*m_omega*0.0) *
                                             m_q0/(4.*utils::pi());
    amrex::GpuComplex<amrex::Real> p_coeff = amrex::GpuComplex<amrex::Real>{0.,1.} *
                                             amrex::exp(amrex::GpuComplex<amrex::Real>{0.,-1.}*m_omega*0.0) *
                                             m_b0/(4.*utils::pi());

    auto& velocity = m_velocity(level);
    auto& density = m_density(level);
    auto& pressure = m_repo.get_field("p")(level);
    auto& gradp = m_repo.get_field("gp")(level);

    density.setVal(m_rho);

    UExact u_exact;
    VExact v_exact;
    WExact w_exact;
    GpxExact gpx_exact;
    GpyExact gpy_exact;
    GpzExact gpz_exact;

    for (amrex::MFIter mfi(velocity); mfi.isValid(); ++mfi) {
        const auto& vbx = mfi.validbox();

        const auto& dx = geom.CellSizeArray();
        const auto& problo = geom.ProbLoArray();
        auto vel = velocity.array(mfi);
        auto gp = gradp.array(mfi);

        amrex::ParallelFor(
            vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const amrex::Real x = problo[0] + (i + 0.5) * dx[0] - cen_x;
                const amrex::Real y = problo[1] + (j + 0.5) * dx[1] - cen_y;
                const amrex::Real z = problo[2] + (k + 0.5) * dx[2] - cen_z;

                vel(i, j, k, 0) = u_exact(lambda, b_coeff, q_coeff, p_coeff, v_coeff, r0, x, y, z, eps);
                vel(i, j, k, 1) = v_exact(lambda, b_coeff, q_coeff, p_coeff, v_coeff, r0, x, y, z, eps);
                vel(i, j, k, 2) = w_exact(lambda, b_coeff, q_coeff, p_coeff, v_coeff, r0, x, y, z, eps);

                gp(i, j, k, 0) = gpx_exact(lambda, b_coeff, q_coeff, p_coeff, v_coeff, r0, x, y, z, eps);
                gp(i, j, k, 1) = gpy_exact(lambda, b_coeff, q_coeff, p_coeff, v_coeff, r0, x, y, z, eps);
                gp(i, j, k, 2) = gpz_exact(lambda, b_coeff, q_coeff, p_coeff, v_coeff, r0, x, y, z, eps);
            });

        // compute initial pressure condition
        const auto& nbx = mfi.nodaltilebox();
        auto pres = pressure.array(mfi);

        amrex::ParallelFor(
            nbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const amrex::Real x = problo[0] + i * dx[0] - cen_x;
                const amrex::Real y = problo[1] + j * dx[1] - cen_y;
                const amrex::Real z = problo[2] + k * dx[2] - cen_z;
                amrex::Real dist = std::sqrt(std::pow(x,2)+std::pow(y,2)+std::pow(z,2));

                if(dist < (r0-eps)) {
                    pres(i, j, k, 0) = 0.0;
                }
                else {
                  amrex::Real v_dot_x = v_coeff[0]*x+v_coeff[1]*y+v_coeff[2]*z;
                  pres(i, j, k, 0) = (p_coeff*v_dot_x/(std::pow(dist,3.)+eps)).real();
                }
            });
    }
}

template <typename T>
amrex::Real StokesOscillatingSphere::compute_error(const Field& field)
{

    amrex::Real error = 0.0;

    const auto eps     = m_eps;
    const auto lambda  = m_lambda;
    const auto r0      = m_r0;
    const auto time    = m_time.new_time();
    const auto v_coeff = m_v_coeff;

    // evaluate translated center of sphere
    const auto cen_x = m_cen[0] - (amrex::exp(
                                  amrex::GpuComplex<amrex::Real>{0.,-1.}*m_omega*time)/m_omega*v_coeff[0]).real();
    const auto cen_y = m_cen[1] - (amrex::exp(
                                  amrex::GpuComplex<amrex::Real>{0.,-1.}*m_omega*time)/m_omega*v_coeff[1]).real();
    const auto cen_z = m_cen[2] - (amrex::exp(
                                  amrex::GpuComplex<amrex::Real>{0.,-1.}*m_omega*time)/m_omega*v_coeff[2]).real();

    // velocity and pressure coefficients
    amrex::GpuComplex<amrex::Real> b_coeff = amrex::GpuComplex<amrex::Real>{0.,1.} *
                                             amrex::exp(amrex::GpuComplex<amrex::Real>{0.,-1.}*m_omega*time) *
                                             m_b0/(8.*utils::pi()*m_nu);
    amrex::GpuComplex<amrex::Real> q_coeff = amrex::GpuComplex<amrex::Real>{0.,1.} *
                                             amrex::exp(amrex::GpuComplex<amrex::Real>{0.,-1.}*m_omega*time) *
                                             m_q0/(4.*utils::pi());
    amrex::GpuComplex<amrex::Real> p_coeff = amrex::GpuComplex<amrex::Real>{0.,1.} *
                                             amrex::exp(amrex::GpuComplex<amrex::Real>{0.,-1.}*m_omega*time) *
                                             m_b0/(4.*utils::pi());

    T f_exact;
    const auto comp = f_exact.m_comp;

    const int nlevels = m_repo.num_active_levels();
    for (int lev = 0; lev < nlevels; ++lev) {

        amrex::iMultiFab level_mask;
        if (lev < nlevels - 1) {
            level_mask = makeFineMask(
                m_mesh.boxArray(lev), m_mesh.DistributionMap(lev),
                m_mesh.boxArray(lev + 1), amrex::IntVect(2), 1, 0);
        } else {
            level_mask.define(
                m_mesh.boxArray(lev), m_mesh.DistributionMap(lev), 1, 0,
                amrex::MFInfo());
            level_mask.setVal(1);
        }

        if (m_sim.has_overset()) {
            for (amrex::MFIter mfi(field(lev)); mfi.isValid(); ++mfi) {
                const auto& vbx = mfi.validbox();

                const auto& iblank_arr =
                    m_repo.get_int_field("iblank_cell")(lev).array(mfi);
                const auto& imask_arr = level_mask.array(mfi);
                amrex::ParallelFor(
                    vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        if (iblank_arr(i, j, k) < 1) imask_arr(i, j, k) = 0;
                    });
            }
        }

        const auto& dx = m_mesh.Geom(lev).CellSizeArray();
        const auto& problo = m_mesh.Geom(lev).ProbLoArray();
        const amrex::Real cell_vol = dx[0] * dx[1] * dx[2];

        const auto& fld = field(lev);
        error += amrex::ReduceSum(
            fld, level_mask, 0,
            [=] AMREX_GPU_HOST_DEVICE(
                amrex::Box const& bx,
                amrex::Array4<amrex::Real const> const& fld_arr,
                amrex::Array4<int const> const& mask_arr) -> amrex::Real {
                amrex::Real err_fab = 0.0;

                amrex::Loop(bx, [=, &err_fab](int i, int j, int k) noexcept {
                    const amrex::Real x = problo[0] + (i + 0.5) * dx[0] - cen_x;
                    const amrex::Real y = problo[1] + (j + 0.5) * dx[1] - cen_y;
                    const amrex::Real z = problo[2] + (k + 0.5) * dx[2] - cen_z;

                    const amrex::Real u = fld_arr(i, j, k, comp);

                    const amrex::Real u_exact = f_exact(lambda, b_coeff, q_coeff, p_coeff, v_coeff, r0, x, y, z, eps);

                    err_fab += cell_vol * mask_arr(i, j, k) * (u - u_exact) *
                               (u - u_exact);
                });
                return err_fab;
            });
    }

    amrex::ParallelDescriptor::ReduceRealSum(error);

    const amrex::Real total_vol = m_mesh.Geom(0).ProbDomain().volume();
    return std::sqrt(error / total_vol);
}

void StokesOscillatingSphere::output_error()
{
    const amrex::Real u_err = compute_error<UExact>(m_velocity);
    const amrex::Real v_err = compute_error<VExact>(m_velocity);
    const amrex::Real w_err = compute_error<WExact>(m_velocity);
    const amrex::Real gpx_err = compute_error<GpxExact>(m_gradp);
    const amrex::Real gpy_err = compute_error<GpyExact>(m_gradp);
    const amrex::Real gpz_err = compute_error<GpzExact>(m_gradp);

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream f;
        f.open(m_output_fname.c_str(), std::ios_base::app);
        f << std::setprecision(12) << std::setw(m_w) << m_time.new_time()
          << std::setw(m_w) << u_err << std::setw(m_w) << v_err
          << std::setw(m_w) << w_err << std::setw(m_w) << gpx_err
          << std::setw(m_w) << gpy_err << std::setw(m_w) << gpz_err
          << std::endl;
        f.close();
    }
}

void StokesOscillatingSphere::post_init_actions() { output_error(); }

void StokesOscillatingSphere::post_advance_work() { output_error(); }

} // namespace sop
} // namespace amr_wind
