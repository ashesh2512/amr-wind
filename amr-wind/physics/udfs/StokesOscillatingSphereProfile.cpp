#include "amr-wind/physics/udfs/StokesOscillatingSphereProfile.H"
#include "amr-wind/core/Field.H"
#include "amr-wind/core/FieldRepo.H"
#include "amr-wind/core/vs/vector.H"
#include "amr-wind/equation_systems/icns/icns.H"

#include "AMReX_ParmParse.H"

namespace amr_wind {
namespace udf {

StokesOscillatingSphereProfile::StokesOscillatingSphereProfile(const Field& fld) : m_op()
{
    AMREX_ALWAYS_ASSERT(fld.name() == pde::ICNS::var_name());
    AMREX_ALWAYS_ASSERT(fld.num_comp() == AMREX_SPACEDIM);

    amrex::ParmParse pp("SOP");
    pp.query("density", m_op.rho);
    pp.queryarr("sphere_center", m_op.cen, 0, AMREX_SPACEDIM);
    pp.query("sphere_radius", m_op.r0);
    pp.queryarr("vel_coeff", m_op.v_coeff, 0, AMREX_SPACEDIM);
    {
        amrex::ParmParse pp("transport");
        pp.query("viscosity", m_op.nu);
    }

    // construct constants related to the analytical solution
    m_op.lambda = amrex::GpuComplex<amrex::Real>{1.,-1.}*m_op.r0*std::sqrt(0.5*std::abs(m_op.omega)/m_op.nu);
    m_op.b0     = 6.*utils::pi()*m_op.nu*m_op.r0*(1.+m_op.lambda+amrex::pow(m_op.lambda,2.)/3.);
    m_op.q0     = -6.*utils::pi()*std::pow(m_op.r0,3.) *
                  (amrex::exp(m_op.lambda)-1.-m_op.lambda-amrex::pow(m_op.lambda,2.)/3.)/m_op.lambda/m_op.lambda;
}

}
}
