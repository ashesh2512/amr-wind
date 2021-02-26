#ifndef STOKESOSCILLATINGSPHEREPROFILE_H
#define STOKESOSCILLATINGSPHEREPROFILE_H

#include "amr-wind/utilities/trig_ops.H"

#include "AMReX_Geometry.H"
#include "AMReX_Gpu.H"
#include "AMReX_GpuComplex.H"

namespace amr_wind {

class Field;

namespace udf {

struct StokesOscillatingSphereProfile
{
    struct DeviceOp
    {
        //! Oscillating frequency
        const amrex::Real eps{1.e-8};

        //! Oscillating frequency
        const amrex::Real omega{2*utils::pi()};

        //! Default viscosity value
        amrex::Real nu{1.0};

        //! Radius of sphere
        amrex::Real r0{0.5};

        //! Initial density value
        amrex::Real rho{1.0};

        //! Center of sphere
        amrex::Vector<amrex::Real> cen{0.0,0.0,0.0};

        //! Default velocity coefficient vector
        amrex::Vector<amrex::Real> v_coeff{1.0,0.0,0.0};

        //! Analytical solution related complex constants
        amrex::GpuComplex<amrex::Real> b0{0.0,0.0};
        amrex::GpuComplex<amrex::Real> lambda{0.0,0.0};
        amrex::GpuComplex<amrex::Real> q0{0.0,0.0};

        AMREX_GPU_DEVICE
        inline void operator()(
            const amrex::IntVect& iv,
            amrex::Array4<amrex::Real> const& field,
            amrex::GeometryData const& geom,
            const amrex::Real t,
            amrex::Orientation,
            const int comp) const
        {
            // evaluate translated center of sphere
             const auto cen_x = cen[0] - (amrex::exp(
                                          amrex::GpuComplex<amrex::Real>{0.,-1.}*omega*t)/omega*v_coeff[0]).real();
             const auto cen_y = cen[1] - (amrex::exp(
                                          amrex::GpuComplex<amrex::Real>{0.,-1.}*omega*t)/omega*v_coeff[1]).real();
             const auto cen_z = cen[2] - (amrex::exp(
                                          amrex::GpuComplex<amrex::Real>{0.,-1.}*omega*t)/omega*v_coeff[2]).real();

             // get relative x, y, z coordinates
             const auto* problo = geom.ProbLo();
             const auto* dx = geom.CellSize();
             amrex::Vector<amrex::Real> x = {
                 problo[0] + (iv[0] + 0.5) * dx[0] - cen_x,
                 problo[1] + (iv[1] + 0.5) * dx[1] - cen_y,
                 problo[2] + (iv[2] + 0.5) * dx[2] - cen_z
             };

             // get distance of point w.r.t. sphere center
             amrex::Real dist = std::sqrt(std::pow(x[0],2)+std::pow(x[1],2)+std::pow(x[2],2));

             // velocity and pressure coefficients
             amrex::GpuComplex<amrex::Real> b_coeff = amrex::GpuComplex<amrex::Real>{0.,1.} *
                                                      amrex::exp(amrex::GpuComplex<amrex::Real>{0.,-1.}*omega*0.0) *
                                                      b0/(8.*utils::pi()*nu);
             amrex::GpuComplex<amrex::Real> q_coeff = amrex::GpuComplex<amrex::Real>{0.,1.} *
                                                      amrex::exp(amrex::GpuComplex<amrex::Real>{0.,-1.}*omega*0.0) *
                                                      q0/(4.*utils::pi());

             // dimensionless distance metrics
             amrex::GpuComplex<amrex::Real> cap_r  = lambda*dist/r0;
             amrex::GpuComplex<amrex::Real> cap_r2 = amrex::pow(cap_r,2.);
             amrex::GpuComplex<amrex::Real> exp_r  = amrex::exp(-cap_r);

             amrex::Real v_dot_x = v_coeff[0]*x[0]+v_coeff[1]*x[1]+v_coeff[2]*x[2];

             // different components of 1st term
             amrex::GpuComplex<amrex::Real> b1 = (2.*exp_r*(1.+1./cap_r+1./cap_r2) - 2./cap_r2) * v_coeff[comp] / (dist+eps);
             amrex::GpuComplex<amrex::Real> b2 = (6./cap_r2 - 2.*exp_r*(1.+3./cap_r+3./cap_r2)) * v_dot_x * x[comp] / (std::pow(dist,3.)+eps);
             amrex::GpuComplex<amrex::Real> t1 = b_coeff * (b1 + b2);

             // different components of 2nd term
             amrex::GpuComplex<amrex::Real> q1 = -exp_r * (1.+cap_r+cap_r2) * v_coeff[comp] / (std::pow(dist,3.)+eps);
             amrex::GpuComplex<amrex::Real> q2 = 3. * exp_r * (1.+cap_r+cap_r2/3.) * v_dot_x * x[comp] / (std::pow(dist,5.)+eps);
             amrex::GpuComplex<amrex::Real> t2 = q_coeff * (q1 + q2);

             // extract real parts
             field(iv[0], iv[1], iv[2], comp) = t1.real() + t2.real();
        }
    };

    using DeviceType = DeviceOp;

    static std::string identifier() { return "StokesOscillatingSphereProfile"; }

    StokesOscillatingSphereProfile(const Field& fld);

    DeviceType device_instance() const { return m_op; }

    DeviceOp m_op;
};

} // namespace udf
} // namespace amr_wind

#endif /* STOKESOSCILLATINGSPHEREPROFILE_H */
