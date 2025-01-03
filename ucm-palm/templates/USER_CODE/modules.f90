!> @file modules.f90
!--------------------------------------------------------------------------------------------------!
! This file is part of the PALM model system.
!
! PALM is free software: you can redistribute it and/or modify it under the terms of the GNU General
! Public License as published by the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! PALM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
! implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
! Public License for more details.
!
! You should have received a copy of the GNU General Public License along with PALM. If not, see
! <http://www.gnu.org/licenses/>.
!
! Copyright 1997-2021 Leibniz Universitaet Hannover
!--------------------------------------------------------------------------------------------------!
!
! Description:
! ------------
!> Definition of global variables
!--------------------------------------------------------------------------------------------------!


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Definition of variables for special advection schemes.
!--------------------------------------------------------------------------------------------------!
 MODULE advection

    USE kinds

    REAL(wp), DIMENSION(:), ALLOCATABLE ::  aex  !< exponential coefficient for the Bott-Chlond advection scheme
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  bex  !< exponential coefficient for the Bott-Chlond advection scheme
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  dex  !< exponential coefficient for the Bott-Chlond advection scheme
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  eex  !< exponential coefficient for the Bott-Chlond advection scheme

    SAVE

 END MODULE advection



!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> The variable in this module is used by multi_agent_system_mod AND netcdf_interface_mod. It must
!> be here to avoid circular dependency.
!> This is a workaround.
!--------------------------------------------------------------------------------------------------!
 MODULE mas_global_attributes

    USE kinds

    INTEGER(iwp) ::  dim_size_agtnum  !< size of agent number dimension for netCDF output

    SAVE

 END MODULE mas_global_attributes


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Definition of all arrays defined on the computational grid.
!--------------------------------------------------------------------------------------------------!
 MODULE arrays_3d

    USE kinds

    REAL(wp), DIMENSION(:), ALLOCATABLE ::  d_exner                         !< ratio of potential and actual temperature
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  ddzu                            !< 1/dzu
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  ddzu_pres                       !< modified ddzu for pressure solver
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  dd2zu                           !< 1/(dzu(k)+dzu(k+1))
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  drho_air                        !< inverse air density profile on the uv grid
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  drho_air_zw                     !< inverse air density profile on the w grid
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  dzu                             !< vertical grid size (u-grid)
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  ddzw                            !< 1/dzw
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  dzw                             !< vertical grid size (w-grid)
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  exner                           !< ratio of actual and potential temperature
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  heatflux_input_conversion       !< conversion factor array for heatflux input
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  heatflux_output_conversion      !< conversion factor array for heatflux output
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  hyp                             !< hydrostatic pressure
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  hyrho                           !< density of air calculated with hydrostatic pressure
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  momentumflux_input_conversion   !< conversion factor array for momentumflux input
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  momentumflux_output_conversion  !< conversion factor array for momentumflux output
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  odf_x                           !< outflow damping factor in x-direction
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  odf_y                           !< outflow damping factor in y-direction
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  ptdf_x                          !< damping factor for potential temperature in
                                                                            !< x-direction
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  ptdf_y                          !< damping factor for potential temperature in
                                                                            !< y-direction
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  pt_init                         !< initial profile of potential temperature
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  q_init                          !< initial profile of total water mixing ratio
                                                                            !< (or total water content with active cloud physics)
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  rdf                             !< rayleigh damping factor for velocity components
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  rdf_sc                          !< rayleigh damping factor for scalar quantities
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  ref_state                       !< reference state of potential temperature
                                                                            !< (and density in case of ocean simulation)
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  rho_air                         !< air density profile on the uv grid
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  rho_air_zw                      !< air density profile on the w grid
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  s_init                          !< initial profile of passive scalar concentration
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  sa_init                         !< initial profile of salinity (ocean)
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  scalarflux_input_conversion     !< conversion factor array for scalarflux input
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  scalarflux_output_conversion    !< conversion factor array for scalarflux output
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  ug                              !< geostrophic wind component in x-direction
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  u_init                          !< initial profile of horizontal velocity component u
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  u_stokes_zu                     !< u-component of Stokes drift velocity at zu levels
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  u_stokes_zw                     !< u-component of Stokes drift velocity at zw levels
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  vg                              !< geostrophic wind component in y-direction
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  v_init                          !< initial profile of horizontal velocity component v
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  v_stokes_zu                     !< v-component of Stokes drift velocity at zu levels
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  v_stokes_zw                     !< v-component of Stokes drift velocity at zw levels
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  waterflux_input_conversion      !< conversion factor array for waterflux input
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  waterflux_output_conversion     !< conversion factor array for waterflux output
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  w_subs                          !< subsidence/ascent velocity
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  x                               !< horizontal grid coordinate of v- and s-grid (in m)
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  xu                              !< horizontal grid coordinate of u-grid (in m)
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  y                               !< horizontal grid coordinate of u- and s-grid (in m)
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  yv                              !< horizontal grid coordinate of v-grid (in m)
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  zu                              !< vertical grid coordinate of u-, v-, and s-grid (in m)
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  zw                              !< vertical grid coordinate of w-grid (in m)


    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  c_u                   !< phase speed of u-velocity component
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  c_v                   !< phase speed of v-velocity component
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  c_w                   !< phase speed of w-velocity component
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  diss_s_diss           !< artificial numerical dissipation flux at south face of grid
                                                                    !< box - TKE dissipation
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  diss_s_e              !< artificial numerical dissipation flux at south face of grid
                                                                    !< box - subgrid-scale TKE
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  diss_s_nc             !< artificial numerical dissipation flux at south face of grid
                                                                    !< box - clouddrop-number concentration
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  diss_s_ng             !< artificial numerical dissipation flux at south face of grid
                                                                    !< box - graupel number concentration
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  diss_s_ni             !< artificial numerical dissipation flux at south face of grid
                                                                    !< box - ice crystal-number concentration
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  diss_s_nr             !< artificial numerical dissipation flux at south face of grid
                                                                    !< box - raindrop-number concentration
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  diss_s_ns             !< artificial numerical dissipation flux at south face of grid
                                                                    !< box - snow-number concentration
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  diss_s_pt             !< artificial numerical dissipation flux at south face of grid
                                                                    !< box - potential temperature
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  diss_s_q              !< artificial numerical dissipation flux at south face of grid
                                                                    !< box - total water mixing ratio
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  diss_s_qc             !< artificial numerical dissipation flux at south face of grid
                                                                    !< box - cloudwater mixing ratio
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  diss_s_qg             !< artificial numerical dissipation flux at south face of grid
                                                                    !< box - graupel mixing ratio
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  diss_s_qi             !< artificial numerical dissipation flux at south face of grid
                                                                    !< box - ice crystal mixing ratio
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  diss_s_qr             !< artificial numerical dissipation flux at south face of grid
                                                                    !< box - rainwater mixing ratio
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  diss_s_qs             !< artificial numerical dissipation flux at south face of grid
                                                                    !< box - snow mixing ratio
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  diss_s_s              !< artificial numerical dissipation flux at south face of grid
                                                                    !< box - passive scalar
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  diss_s_sa             !< artificial numerical dissipation flux at south face of grid
                                                                    !< box - salinity
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  diss_s_u              !< artificial numerical dissipation flux at south face of grid
                                                                    !< box - u-component
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  diss_s_v              !< artificial numerical dissipation flux at south face of grid
                                                                    !< box - v-component
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  diss_s_w              !< artificial numerical dissipation flux at south face of grid
                                                                    !< box - w-component
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  dzu_mg                !< vertical grid size (u-grid) for multigrid pressure solver
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  dzw_mg                !< vertical grid size (w-grid) for multigrid pressure solver
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  flux_s_diss           !< 6th-order advective flux at south face of grid box -
                                                                    !< TKE dissipation
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  flux_s_e              !< 6th-order advective flux at south face of grid box -
                                                                    !< subgrid-scale TKE
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  flux_s_nc             !< 6th-order advective flux at south face of grid box -
                                                                    !< clouddrop-number concentration
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  flux_s_ng             !< 6th-order advective flux at south face of grid box -
                                                                    !< graupel-number concentration
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  flux_s_ni             !< 6th-order advective flux at south face of grid box -
                                                                    !< icecrystal-number concentration
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  flux_s_nr             !< 6th-order advective flux at south face of grid box -
                                                                    !< raindrop-number concentration
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  flux_s_ns             !< 6th-order advective flux at south face of grid box -
                                                                    !< graupel-number concentration
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  flux_s_pt             !< 6th-order advective flux at south face of grid box -
                                                                    !< potential temperature
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  flux_s_q              !< 6th-order advective flux at south face of grid box -
                                                                    !< total water mixing ratio
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  flux_s_qc             !< 6th-order advective flux at south face of grid box -
                                                                    !< cloudwater mixing ratio
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  flux_s_qg             !< 6th-order advective flux at south face of grid box -
                                                                    !< graupel mixing ratio
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  flux_s_qi             !< 6th-order advective flux at south face of grid box -
                                                                    !< ice crystal mixing ratio
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  flux_s_qr             !< 6th-order advective flux at south face of grid box -
                                                                    !< rainwater mixing ratio
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  flux_s_qs             !< 6th-order advective flux at south face of grid box -
                                                                    !< snow mixing ratio
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  flux_s_s              !< 6th-order advective flux at south face of grid box -
                                                                    !< passive scalar
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  flux_s_sa             !< 6th-order advective flux at south face of grid box -
                                                                    !< salinity
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  flux_s_u              !< 6th-order advective flux at south face of grid box -
                                                                    !< u-component
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  flux_s_v              !< 6th-order advective flux at south face of grid box -
                                                                    !< v-component
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  flux_s_w              !< 6th-order advective flux at south face of grid box -
                                                                    !< w-component
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  f1_mg                 !< grid factor used in right hand side of Gauss-Seidel equation
                                                                    !< (multigrid)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  f2_mg                 !< grid factor used in right hand side of Gauss-Seidel equation
                                                                    !< (multigrid)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  f3_mg                 !< grid factor used in right hand side of Gauss-Seidel equation
                                                                    !< (multigrid)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  mean_inflow_profiles  !< used for turbulent inflow (non-cyclic boundary conditions)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  precipitation_amount  !< precipitation amount due to gravitational settling
                                                                    !< (bulk microphysics)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  pt_slope_ref          !< potential temperature in rotated coordinate system
                                                                    !< (in case of sloped surface)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  rho_air_mg            !< air density profiles on the uv grid for multigrid
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  rho_air_zw_mg         !< air density profiles on the w grid for multigrid
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  total_2d_a            !< horizontal array to store the total domain data, used for
                                                                    !< atmosphere-ocean coupling (atmosphere data)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  total_2d_o            !< horizontal array to store the total domain data, used for
                                                                    !< atmosphere-ocean coupling (ocean data)

    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  d           !< divergence
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  de_dx       !< gradient of sgs tke in x-direction (lpm)
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  de_dy       !< gradient of sgs tke in y-direction (lpm)
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  de_dz       !< gradient of sgs tke in z-direction (lpm)
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  diss_l_diss !< artificial numerical dissipation flux at left face of grid box -
                                                            !< TKE dissipation
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  diss_l_e    !< artificial numerical dissipation flux at left face of grid box -
                                                            !< subgrid-scale TKE
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  diss_l_nc   !< artificial numerical dissipation flux at left face of grid box -
                                                            !< clouddrop-number concentration
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  diss_l_ng   !< artificial numerical dissipation flux at left face of grid box -
                                                            !< graupel-number concentration
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  diss_l_ni   !< artificial numerical dissipation flux at left face of grid box -
                                                            !< ice crystal-number concentration
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  diss_l_nr   !< artificial numerical dissipation flux at left face of grid box -
                                                            !< raindrop-number concentration
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  diss_l_ns   !< artificial numerical dissipation flux at left face of grid box -
                                                            !< snow-number concentration
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  diss_l_pt   !< artificial numerical dissipation flux at left face of grid box -
                                                            !< potential temperature
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  diss_l_q    !< artificial numerical dissipation flux at left face of grid box -
                                                            !< total water mixing ratio
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  diss_l_qc   !< artificial numerical dissipation flux at left face of grid box -
                                                            !< cloudwater
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  diss_l_qg   !< artificial numerical dissipation flux at left face of grid box -
                                                            !< graupel
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  diss_l_qi   !< artificial numerical dissipation flux at left face of grid box -
                                                            !< ice crystal
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  diss_l_qr   !< artificial numerical dissipation flux at left face of grid box -
                                                            !< rainwater
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  diss_l_qs   !< artificial numerical dissipation flux at left face of grid box -
                                                            !< snow
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  diss_l_s    !< artificial numerical dissipation flux at left face of grid box -
                                                            !< passive scalar
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  diss_l_sa   !< artificial numerical dissipation flux at left face of grid box -
                                                            !< salinity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  diss_l_u    !< artificial numerical dissipation flux at left face of grid box -
                                                            !< u-component
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  diss_l_v    !< artificial numerical dissipation flux at left face of grid box -
                                                            !< v-component
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  diss_l_w    !< artificial numerical dissipation flux at left face of grid box -
                                                            !< w-component
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  flux_l_diss !< 6th-order advective flux at south face of grid box - TKE dissipation
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  flux_l_e    !< 6th-order advective flux at south face of grid box - subgrid-scale
                                                            !< TKE
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  flux_l_nc   !< 6th-order advective flux at south face of grid box - clouddrop-number
                                                            !< concentration
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  flux_l_ng   !< 6th-order advective flux at south face of grid box -
                                                            !< graupel-number concentration
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  flux_l_ni   !< 6th-order advective flux at south face of grid box -
                                                            !< ice crystal-number concentration
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  flux_l_nr   !< 6th-order advective flux at south face of grid box - raindrop-number
                                                            !< concentration
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  flux_l_ns   !< 6th-order advective flux at south face of grid box - snow-number concentration
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  flux_l_pt   !< 6th-order advective flux at south face of grid box - potential
                                                            !< temperature
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  flux_l_q    !< 6th-order advective flux at south face of grid box - mixing ratio
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  flux_l_qc   !< 6th-order advective flux at south face of grid box - cloudwater
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  flux_l_qg   !< 6th-order advective flux at south face of grid box - graupel
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  flux_l_qi   !< 6th-order advective flux at south face of grid box - ice crystal
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  flux_l_qr   !< 6th-order advective flux at south face of grid box - rainwater
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  flux_l_qs   !< 6th-order advective flux at south face of grid box - snow
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  flux_l_s    !< 6th-order advective flux at south face of grid box - passive scalar
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  flux_l_sa   !< 6th-order advective flux at south face of grid box - salinity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  flux_l_u    !< 6th-order advective flux at south face of grid box - u-component
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  flux_l_v    !< 6th-order advective flux at south face of grid box - v-component
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  flux_l_w    !< 6th-order advective flux at south face of grid box - w-component
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  kh  !< eddy diffusivity for heat
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  km  !< eddy diffusivity for momentum
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  prr         !< total precipitation rate (all phases)
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  prr_cloud   !< cloud sedimentation rate
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  prr_graupel !< graupel precipitation rate
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  prr_ice     !< ice precipitation rate
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  prr_rain    !< rain precipitation rate
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  prr_snow    !< snow precipitation rate
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  p_loc       !< local array in multigrid/sor solver containing the pressure which is
                                                            !< iteratively advanced in each iteration step
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  tend        !< tendency field (time integration)

    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  diss_1  !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  diss_2  !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  diss_3  !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  e_1     !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  e_2     !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  e_3     !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  p       !< pointer: perturbation pressure
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  prho_1  !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  nc_1    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  nc_2    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  nc_3    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  ng_1    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  ng_2    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  ng_3    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  ni_1    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  ni_2    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  ni_3    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  nr_1    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  nr_2    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  nr_3    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  ns_1    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  ns_2    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  ns_3    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  pt_1    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  pt_2    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  pt_3    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  q_1     !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  q_2     !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  q_3     !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qc_1    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qc_2    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qc_3    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qf_1    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qg_1    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qg_2    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qg_3    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qi_1    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qi_2    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qi_3    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  ql_v    !< pointer: volume of liquid water
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  ql_vp   !< pointer: liquid water weighting factor
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  ql_1    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  ql_2    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qr_1    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qr_2    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qr_3    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qs_1    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qs_2    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qs_3    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  rho_1   !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  s_1     !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  s_2     !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  s_3     !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  sa_1    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  sa_2    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  sa_3    !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  u_1     !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  u_2     !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  u_3     !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  v_1     !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  v_2     !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  v_3     !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  vpt_1   !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  w_1     !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  w_2     !< pointer for swapping of timelevels for respective quantity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  w_3     !< pointer for swapping of timelevels for respective quantity

    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  diss       !< pointer: TKE dissipation
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  diss_p     !< pointer: prognostic value of TKE dissipation
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  e          !< pointer: subgrid-scale turbulence kinetic energy (sgs tke)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  e_p        !< pointer: prognostic value of sgs tke
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  nc         !< pointer: cloud drop number density
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  nc_p       !< pointer: prognostic value of cloud drop number density
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  ng         !< pointer: graupel number density
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  ng_p       !< pointer: prognostic value of graupel number density
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  ni         !< pointer: ice crystal number density
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  ni_p       !< pointer: prognostic value of ice crystal number density
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  nr         !< pointer: rain drop number density
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  nr_p       !< pointer: prognostic value of rain drop number density
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  ns         !< pointer: snow number density
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  ns_p       !< pointer: prognostic value of snow number density
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  prho       !< pointer: potential density
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  pt         !< pointer: potential temperature
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  pt_p       !< pointer: prognostic value of potential temperature
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  q          !< pointer: mixing ratio
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  q_p        !< pointer: prognostic value of mixing ratio
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  qc         !< pointer: cloud water content
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  qc_p       !< pointer: prognostic value cloud water content
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  qf         !< pointer: frozen water content
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  qg         !< pointer: graupel water content
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  qg_p       !< pointer: prognostic value graupel water content
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  qi         !< pointer: ice crystal content
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  qi_p       !< pointer: prognostic value ice crystal content
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  ql         !< pointer: liquid water content
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  ql_c       !< pointer: change in liquid water content due to
                                                                   !< condensation/evaporation during last time step
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  qr         !< pointer: rain water content
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  qr_p       !< pointer: prognostic value of rain water content
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  qs         !< pointer: rain water content
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  qs_p       !< pointer: prognostic value of rain water content
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  rho_ocean  !< pointer: density of ocean
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  s          !< pointer: passive scalar
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  s_p        !< pointer: prognostic value of passive scalar
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  sa         !< pointer: ocean salinity
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  sa_p       !< pointer: prognostic value of ocean salinity
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  tdiss_m    !< pointer: weighted tendency of diss for previous sub-timestep
                                                                   !< (Runge-Kutta)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  te_m       !< pointer: weighted tendency of e for previous sub-timestep
                                                                   !< (Runge-Kutta)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  tnc_m      !< pointer: weighted tendency of nc for previous sub-timestep
                                                                   !< (Runge-Kutta)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  tng_m      !< pointer: weighted tendency of ng for previous sub-timestep (Runge-Kutta)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  tni_m      !< pointer: weighted tendency of ni for previous sub-timestep
                                                                   !< (Runge-Kutta)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  tnr_m      !< pointer: weighted tendency of nr for previous sub-timestep
                                                                   !< (Runge-Kutta)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  tns_m      !< pointer: weighted tendency of ns for previous sub-timestep (Runge-Kutta)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  tpt_m      !< pointer: weighted tendency of pt for previous sub-timestep
                                                                   !< (Runge-Kutta)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  tq_m       !< pointer: weighted tendency of q for previous sub-timestep
                                                                   !< (Runge-Kutta)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  tqc_m      !< pointer: weighted tendency of qc for previous sub-timestep
                                                                   !< (Runge-Kutta)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  tqg_m      !< pointer: weighted tendency of qg for previous sub-timestep (Runge-Kutta)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  tqi_m      !< pointer: weighted tendency of qi for previous sub-timestep
                                                                   !< (Runge-Kutta)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  tqr_m      !< pointer: weighted tendency of qr for previous sub-timestep
                                                                   !< (Runge-Kutta)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  tqs_m      !< pointer: weighted tendency of qs for previous sub-timestep (Runge-Kutta)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  ts_m       !< pointer: weighted tendency of s for previous sub-timestep
                                                                   !< (Runge-Kutta)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  tsa_m      !< pointer: weighted tendency of sa for previous sub-timestep
                                                                   !< (Runge-Kutta)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  tu_m       !< pointer: weighted tendency of u for previous sub-timestep
                                                                   !< (Runge-Kutta)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  tv_m       !< pointer: weighted tendency of v for previous sub-timestep
                                                                   !< (Runge-Kutta)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  tw_m       !< pointer: weighted tendency of w for previous sub-timestep
                                                                   !< (Runge-Kutta)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  u          !< pointer: horizontal velocity component u (x-direction)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  u_p        !< pointer: prognostic value of u
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  v          !< pointer: horizontal velocity component v (y-direction)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  v_p        !< pointer: prognostic value of v
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  vpt        !< pointer: virtual potential temperature
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  w          !< pointer: vertical velocity component w (z-direction)
    REAL(wp), DIMENSION(:,:,:), POINTER, CONTIGUOUS ::  w_p        !< pointer: prognostic value of w

    SAVE

 END MODULE arrays_3d


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Definition of variables needed for time-averaging of 2d/3d data.
!--------------------------------------------------------------------------------------------------!
 MODULE averaging

    USE kinds

    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  ghf_av          !< avg. ground heat flux
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  lwp_av          !< avg. liquid water path
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  ol_av           !< avg. Obukhov length
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  pres_drag_x_av  !< avg. pressure drag on yz-surfaces
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  pres_drag_y_av  !< avg. pressure drag on xz-surfaces
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  qsurf_av        !< avg. surface mixing ratio
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  qsws_av         !< avg. surface moisture flux
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  r_a_av          !< avg. resistance
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  ssurf_av        !< avg. surface passive scalar
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  ssws_av         !< avg. surface scalar flux
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  shf_av          !< avg. surface heat flux
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  tsurf_av        !< avg. surface temperature
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  ts_av           !< avg. characteristic temperature scale
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  us_av           !< avg. friction velocity
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  z0_av           !< avg. roughness length for momentum
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  z0h_av          !< avg. roughness length for heat
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  z0q_av          !< avg. roughness length for moisture

    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  diss_av       !< avg. tke dissipation rate
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  e_av          !< avg. subgrid-scale tke
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  kh_av         !< avg. eddy diffusivity for heat
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  km_av         !< avg. eddy diffusivity for momentum
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  lpt_av        !< avg. liquid water potential temperature
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  nc_av         !< avg. cloud drop number density
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  ng_av         !< avg. graupel drop number density
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  ni_av         !< avg. ice crystal number density
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  nr_av         !< avg. rain drop number density
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  ns_av         !< avg. snow drop number density
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  p_av          !< avg. perturbation pressure
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  pc_av         !< avg. particle/droplet concentration
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  pr_av         !< avg. particle/droplet radius
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  prr_av        !< avg. precipitation rate (all phases)
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  prr_cloud_av  !< avg. cloud sedimentation rate
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  prr_graupel_av!< avg. graupel precipitation rate
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  prr_ice_av    !< avg. ice precipitation rate
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  prr_rain_av   !< avg. rain precipitation rate
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  prr_snow_av   !< avg. snow precipitation rate
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  pt_av         !< avg. potential temperature
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  q_av          !< avg. mixing ratio
                                                                      !< (or total water content with active cloud physics)
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qc_av         !< avg. cloud water content
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qg_av         !< avg. graupel water content
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qi_av         !< avg. ice crystal content
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  ql_av         !< avg. liquid water content
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  ql_c_av       !< avg. change in liquid water content due to
                                                                      !< condensation/evaporation during last time step
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  ql_v_av       !< avg. volume of liquid water
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  ql_vp_av      !< avg. liquid water weighting factor
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qr_av         !< avg. rain water content
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qs_av         !< avg. snow water content
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  qv_av         !< avg. water vapor content (mixing ratio)
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  rho_ocean_av  !< avg. ocean density
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  s_av          !< avg. passive scalar
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  sa_av         !< avg. salinity
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  u_av          !< avg. horizontal velocity component u
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  v_av          !< avg. horizontal velocity component v
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  vpt_av        !< avg. virtual potential temperature
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  w_av          !< avg. vertical velocity component

 END MODULE averaging


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Definition of parameters for program control
!--------------------------------------------------------------------------------------------------!
 MODULE control_parameters

    USE kinds

    TYPE file_status
       LOGICAL ::  opened         !< file is currently open
       LOGICAL ::  opened_before  !< file is currently closed, but has been openend before
    END TYPE file_status

    INTEGER(iwp), PARAMETER ::  fl_max = 500              !< maximum number of virtual-flight measurements
    INTEGER,      PARAMETER ::  mask_xyz_dimension = 100  !< limit of mask dimensions (100 points in each direction)
    INTEGER,      PARAMETER ::  max_masks = 300           !< maximum number of masks
    INTEGER(iwp), PARAMETER ::  var_fl_max = 20           !< maximum number of different sampling variables in virtual flight
                                                          !< measurements
    INTEGER(iwp), PARAMETER ::  varnamelength = 30        !< length of output variable names

    REAL(wp), PARAMETER ::  output_fill_value = -999999.0_wp

    TYPE(file_status), DIMENSION(200+2*max_masks) ::                &  !< indicates if file is open or if it has been opened before
                             openfile = file_status(.FALSE.,.FALSE.)

    CHARACTER (LEN=1)    ::  cycle_mg = 'w'                               !< namelist parameter (see documentation)
    CHARACTER (LEN=1)    ::  timestep_reason = ' '                        !< 'A'dvection or 'D'iffusion criterion, written to
                                                                          !< RUN_CONTROL file
    CHARACTER (LEN=5)    ::  run_zone = ' '                               !< time zone of simulation run
    CHARACTER (LEN=8)    ::  coupling_char = ''                           !< appended to filenames in coupled or nested runs
                                                                          !< ('_O': ocean PE,
                                                                          !< '_NV': vertically nested atmosphere PE, '_N##': PE of
                                                                          !< nested domain ##
    CHARACTER (LEN=8)    ::  run_time = ' '                               !< time of simulation run
    CHARACTER (LEN=9)    ::  simulated_time_chr                           !< simulated time, printed to RUN_CONTROL file
    CHARACTER (LEN=10)   ::  run_date = ' '                               !< date of simulation run
    CHARACTER (LEN=11)   ::  topography_grid_convention = ' '             !< namelist parameter
    CHARACTER (LEN=26)   ::  version_string = ' '                         !< PALM version
    CHARACTER (LEN=12)   ::  user_interface_required_revision = '1.0'     !< required user-interface revision number (only major number is checked)
    CHARACTER (LEN=16)   ::  conserve_volume_flow_mode = 'initial_profiles'  !< namelist parameter
    CHARACTER (LEN=16)   ::  loop_optimization = 'cache'                  !< namelist parameter
    CHARACTER (LEN=16)   ::  momentum_advec = 'ws-scheme'                 !< namelist parameter
    CHARACTER (LEN=16)   ::  psolver = 'poisfft'                          !< namelist parameter
    CHARACTER (LEN=16)   ::  scalar_advec = 'ws-scheme'                   !< namelist parameter
    CHARACTER (LEN=20)   ::  approximation = 'boussinesq'                 !< namelist parameter
    CHARACTER (LEN=20)   ::  bc_e_b = 'neumann'                           !< namelist parameter
    CHARACTER (LEN=20)   ::  bc_lr = 'cyclic'                             !< namelist parameter
    CHARACTER (LEN=20)   ::  bc_ns = 'cyclic'                             !< namelist parameter
    CHARACTER (LEN=20)   ::  bc_p_b = 'neumann'                           !< namelist parameter
    CHARACTER (LEN=20)   ::  bc_p_t = 'default'                           !< namelist parameter
    CHARACTER (LEN=20)   ::  bc_pt_b = 'dirichlet'                        !< namelist parameter
    CHARACTER (LEN=20)   ::  bc_pt_t = 'initial_gradient'                 !< namelist parameter
    CHARACTER (LEN=20)   ::  bc_q_b = 'dirichlet'                         !< namelist parameter
    CHARACTER (LEN=20)   ::  bc_q_t = 'neumann'                           !< namelist parameter
    CHARACTER (LEN=20)   ::  bc_s_b = 'dirichlet'                         !< namelist parameter
    CHARACTER (LEN=20)   ::  bc_s_t = 'initial_gradient'                  !< namelist parameter
    CHARACTER (LEN=20)   ::  bc_uv_b = 'dirichlet'                        !< namelist parameter
    CHARACTER (LEN=20)   ::  bc_uv_t = 'dirichlet'                        !< namelist parameter
    CHARACTER (LEN=20)   ::  dissipation_1d = 'detering'                  !< namelist parameter
    CHARACTER (LEN=20)   ::  fft_method = 'temperton-algorithm'           !< namelist parameter
    CHARACTER (LEN=20)   ::  mixing_length_1d = 'blackadar'               !< namelist parameter
    CHARACTER (LEN=20)   ::  random_generator = 'random-parallel'         !< namelist parameter
    CHARACTER (LEN=20)   ::  reference_state = 'initial_profile'          !< namelist parameter
    CHARACTER (LEN=20)   ::  restart_data_format = 'mpi_shared_memory'    !< namelist parameter
    CHARACTER (LEN=20)   ::  restart_data_format_input = 'undefined'      !< namelist parameter
    CHARACTER (LEN=20)   ::  restart_data_format_output = 'undefined'     !< namelist parameter
    CHARACTER (LEN=20)   ::  timestep_scheme = 'runge-kutta-3'            !< namelist parameter
    CHARACTER (LEN=20)   ::  turbulence_closure = '1.5-order'             !< namelist parameter
    CHARACTER (LEN=23)   ::  origin_date_time = '2019-06-21 12:00:00 +00' !< date and time to be simulated
    CHARACTER (LEN=40)   ::  flux_input_mode = 'application-specific'     !< type of flux input: dynamic or kinematic
    CHARACTER (LEN=40)   ::  flux_output_mode = 'application-specific'    !< type of flux output: dynamic or kinematic
    CHARACTER (LEN=40)   ::  topography = 'flat'                          !< namelist parameter
    CHARACTER (LEN=64)   ::  host = '????'                                !< configuration identifier as given by palmrun option -c,
                                                                          !< ENVPAR namelist parameter provided by palmrun
    CHARACTER (LEN=80)   ::  log_message                                  !< user-defined message for debugging (sse data_log.f90)
    CHARACTER (LEN=80)   ::  run_identifier                               !< run identifier as given by palmrun option -r, ENVPAR
                                                                          !< namelist parameter provided by palmrun
    CHARACTER (LEN=100)  ::  initializing_actions = ' '                   !< namelist parameter
    CHARACTER (LEN=100)  ::  restart_string = ' '                         !< for storing strings in case of writing/reading restart
                                                                          !< data
    CHARACTER (LEN=210)  ::  run_description_header                       !< string containing diverse run informations as run
                                                                          !< identifier, coupling mode, host, ensemble number, run
                                                                          !< date and time
    CHARACTER (LEN=1000) ::  debug_string = ' '                           !<.....
    CHARACTER (LEN=1000) ::  message_string = ' '                         !< dynamic string for error message output

    CHARACTER (LEN=varnamelength), DIMENSION(300) ::  data_output_pr_user = ' '  !< namelist parameter
    CHARACTER (LEN=varnamelength), DIMENSION(500) ::  data_output_pr = ' '       !< namelist parameter
    CHARACTER (LEN=varnamelength), DIMENSION(500) ::  data_output = ' '          !< namelist parameter
    CHARACTER (LEN=varnamelength), DIMENSION(500) ::  data_output_user = ' '     !< namelist parameter
    CHARACTER (LEN=varnamelength), DIMENSION(500) ::  doav = ' '                 !< label array for multi-dimensional,
                                                                                 !< averaged output quantities

    CHARACTER (LEN=varnamelength), DIMENSION(max_masks,100) ::  data_output_masks = ' '       !< namelist parameter
    CHARACTER (LEN=varnamelength), DIMENSION(max_masks,100) ::  data_output_masks_user = ' '  !< namelist parameter
    CHARACTER (LEN=varnamelength), DIMENSION(0:1,500)       ::  do2d = ' '                    !< label array for 2d output
                                                                                              !< quantities
    CHARACTER (LEN=varnamelength), DIMENSION(0:1,500)       ::  do3d = ' '                    !< label array for 3d output
                                                                                              !< quantities

    CHARACTER (LEN=varnamelength), DIMENSION(max_masks,0:1,100) ::  domask = ' ' !< label array for multi-dimensional,
                                                                                 !< masked output quantities

    INTEGER(iwp) ::  abort_mode = 1                    !< abort condition (nested runs)
    INTEGER(iwp) ::  agt_time_count = 0                !< number of output intervals for agent data output
    INTEGER(iwp) ::  average_count_pr = 0              !< number of samples in vertical-profile output
    INTEGER(iwp) ::  average_count_3d = 0              !< number of samples in 3d output
    INTEGER(iwp) ::  current_timestep_number = 0       !< current timestep number, printed to RUN_CONTROL file
    INTEGER(iwp) ::  dist_range = 0                    !< switch for steering the horizontal disturbance range, 1: inflow
                                                       !< disturbances in case of non-cyclic horizontal BC, 0: otherwise
    INTEGER(iwp) ::  disturbance_level_ind_b           !< lowest grid index where flow disturbance is applied
    INTEGER(iwp) ::  disturbance_level_ind_t           !< highest grid index where flow disturbance is applied
    INTEGER(iwp) ::  doav_n = 0                        !< number of 2d/3d output quantities subject to time averaging
    INTEGER(iwp) ::  dopr_n = 0                        !< number of profile output quantities subject to time averaging
    INTEGER(iwp) ::  dopr_time_count = 0               !< number of output intervals for profile output
    INTEGER(iwp) ::  dots_time_count = 0               !< number of output intervals for timeseries output
    INTEGER(iwp) ::  dp_level_ind_b = 0                !< lowest grid index for external pressure gradient forcing
    INTEGER(iwp) ::  ensemble_member_nr = 0            !< namelist parameter
    INTEGER(iwp) ::  gamma_mg                          !< switch for steering the multigrid cycle: 1: v-cycle, 2: w-cycle
    INTEGER(iwp) ::  gathered_size                     !< number of total domain grid points of the grid level which is gathered on
                                                       !< PE0 (multigrid solver)
    INTEGER(iwp) ::  grid_level                        !< current grid level handled in the multigrid solver
    INTEGER(iwp) ::  ibc_e_b                           !< integer flag for bc_e_b
    INTEGER(iwp) ::  ibc_p_b                           !< integer flag for bc_p_b
    INTEGER(iwp) ::  ibc_p_t                           !< integer flag for bc_p_t
    INTEGER(iwp) ::  ibc_pt_b                          !< integer flag for bc_pt_b
    INTEGER(iwp) ::  ibc_pt_t                          !< integer flag for bc_pt_t
    INTEGER(iwp) ::  ibc_q_b                           !< integer flag for bc_q_b
    INTEGER(iwp) ::  ibc_q_t                           !< integer flag for bc_q_t
    INTEGER(iwp) ::  ibc_s_b                           !< integer flag for bc_s_b
    INTEGER(iwp) ::  ibc_s_t                           !< integer flag for bc_s_t
    INTEGER(iwp) ::  ibc_uv_b                          !< integer flag for bc_uv_b
    INTEGER(iwp) ::  ibc_uv_t                          !< integer flag for bc_uv_t
    INTEGER(iwp) ::  inflow_disturbance_begin = -1     !< namelist parameter
    INTEGER(iwp) ::  inflow_disturbance_end = -1       !< namelist parameter
    INTEGER(iwp) ::  intermediate_timestep_count       !< number of current Runge-Kutta substep
    INTEGER(iwp) ::  intermediate_timestep_count_max   !< maximum number of Runge-Kutta substeps
    INTEGER(iwp) ::  io_group = 0                      !< I/O group to which the PE belongs (= #PE / io_blocks)
    INTEGER(iwp) ::  io_blocks = 1                     !< number of blocks for which I/O is done in sequence (total number of PEs /
                                                       !< maximum_parallel_io_streams)
    INTEGER(iwp) ::  iran = -1234567                   !< integer random number used for flow disturbances
    INTEGER(iwp) ::  length = 0                        !< integer that specifies the length of a string in case of writing/reading
                                                       !< restart data
    INTEGER(iwp) ::  masks = 0                         !< counter for number of masked output quantities
    INTEGER(iwp) ::  maximum_grid_level                !< number of grid levels that the multigrid solver is using
    INTEGER(iwp) ::  maximum_parallel_io_streams = -1  !< maximum number of parallel io streams that the underlying parallel file
                                                       !< system allows, set with palmrun option -w, ENVPAR namelist parameter, provided by palmrun
    INTEGER(iwp) ::  max_pr_salsa = 0                  !< number of salsa profiles (must not change within a job chain)
    INTEGER(iwp) ::  max_pr_user = 0                   !< number of user-defined profiles (must not change within a job chain)
    INTEGER(iwp) ::  max_pr_user_tmp = 0               !< number of user-defined profiles that is temporary stored to check it
                                                       !< against max_pr_user in case of restarts
    INTEGER(iwp) ::  mgcycles = 0                      !< number of multigrid cycles that the multigrid solver has actually carried
                                                       !< out
    INTEGER(iwp) ::  mg_cycles = 4                     !< namelist parameter
    INTEGER(iwp) ::  mg_switch_to_pe0_level = -1       !< namelist parameter
    INTEGER(iwp) ::  ngsrb = 2                         !< namelist parameter
    INTEGER(iwp) ::  nr_timesteps_this_run = 0         !< number of timesteps (cpu time measurements)
    INTEGER(iwp) ::  nsor = 20                         !< namelist parameter
    INTEGER(iwp) ::  nsor_ini = 100                    !< namelist parameter
    INTEGER(iwp) ::  n_sor                             !< number of iterations to be used in SOR-scheme
    INTEGER(iwp) ::  normalizing_region = 0            !< namelist parameter
    INTEGER(iwp) ::  num_mean_inflow_profiles = 7      !< number of mean inflow profiles in case of turbulent inflow
    INTEGER(iwp) ::  num_leg=0                         !< number of different legs in virtual flight measurements
    INTEGER(iwp) ::  num_var_fl                        !< number of sampling/output variables in virtual flight measurements
    INTEGER(iwp) ::  num_var_fl_user=0                 !< number of user-defined sampling/output variables in virtual flight
                                                       !< measurements
    INTEGER(iwp) ::  number_dz                         !< number of user-specified dz values
    INTEGER(iwp) ::  number_stretch_level_start        !< number of user-specified start levels for stretching
    INTEGER(iwp) ::  number_stretch_level_end          !< number of user-specified end levels for stretching
    INTEGER(iwp) ::  nz_do3d = -9999                   !< namelist parameter
    INTEGER(iwp) ::  prt_time_count = 0                !< number of output intervals for particle data output
    INTEGER(iwp) ::  runnr = 0                         !< number of run in job chain
    INTEGER(iwp) ::  subdomain_size                    !< number of grid points in (3d) subdomain including ghost points
    INTEGER(iwp) ::  symmetry_flag = 0                 !< flag for sterring the symmetric behavior of the bottom and top boundary
    INTEGER(iwp) ::  terminate_coupled = 0             !< switch for steering termination in case of coupled runs
    INTEGER(iwp) ::  timestep_count = 0                !< number of timesteps carried out since the beginning of the initial run
    INTEGER(iwp) ::  y_shift = 0                       !< namelist parameter

    INTEGER(iwp) ::  dist_nxl(0:1)                               !< left boundary of disturbance region
    INTEGER(iwp) ::  dist_nxr(0:1)                               !< right boundary of disturbance region
    INTEGER(iwp) ::  dist_nyn(0:1)                               !< north boundary of disturbance region
    INTEGER(iwp) ::  dist_nys(0:1)                               !< south boundary of disturbance region
    INTEGER(iwp) ::  do2d_no(0:1) = 0                            !< number of 2d output quantities
    INTEGER(iwp) ::  do2d_xy_time_count(0:1) = 0                 !< number of output intervals for 2d data (xy)
    INTEGER(iwp) ::  do2d_xz_time_count(0:1) = 0                 !< number of output intervals for 2d data (xz)
    INTEGER(iwp) ::  do2d_yz_time_count(0:1) = 0                 !< number of output intervals for 2d data (yz)
    INTEGER(iwp) ::  do3d_no(0:1) = 0                            !< number of 3d output quantities
    INTEGER(iwp) ::  do3d_time_count(0:1) = 0                    !< number of output intervals for 3d data
    INTEGER(iwp) ::  domask_no(max_masks,0:1) = 0                !< number of masked output quantities
    INTEGER(iwp) ::  domask_time_count(max_masks,0:1)            !< number of output intervals for masked data
    INTEGER(iwp) ::  dz_stretch_level_end_index(9)               !< vertical grid level index until which the vertical grid spacing
                                                                 !< is stretched
    INTEGER(iwp) ::  dz_stretch_level_start_index(9)             !< vertical grid level index above which the vertical grid spacing
                                                                 !< is stretched
    INTEGER(iwp) ::  mask_size(max_masks,3) = -1                 !< size of mask array per mask and dimension (for netcdf output)
    INTEGER(iwp) ::  mask_size_l(max_masks,3) = -1               !< subdomain size of mask array per mask and dimension
                                                                 !< (for netcdf output)
    INTEGER(iwp) ::  mask_start_l(max_masks,3) = -1              !< subdomain start index of mask array (for netcdf output)
    INTEGER(iwp) ::  pt_vertical_gradient_level_ind(10) = -9999  !< grid index values of pt_vertical_gradient_level(s)
    INTEGER(iwp) ::  q_vertical_gradient_level_ind(10) = -9999   !< grid index values of q_vertical_gradient_level(s)
    INTEGER(iwp) ::  s_vertical_gradient_level_ind(10) = -9999   !< grid index values of s_vertical_gradient_level(s)
    INTEGER(iwp) ::  section(100,3)                              !< collective array for section_xy/xz/yz
    INTEGER(iwp) ::  section_xy(100) = -9999                     !< namelist parameter
    INTEGER(iwp) ::  section_xz(100) = -9999                     !< namelist parameter
    INTEGER(iwp) ::  section_yz(100) = -9999                     !< namelist parameter
    INTEGER(iwp) ::  ug_vertical_gradient_level_ind(10) = -9999  !< grid index values of ug_vertical_gradient_level(s)
    INTEGER(iwp) ::  vg_vertical_gradient_level_ind(10) = -9999  !< grid index values of vg_vertical_gradient_level(s)
    INTEGER(iwp) ::  subs_vertical_gradient_level_i(10) = -9999  !< grid index values of subs_vertical_gradient_level(s)
    INTEGER(iwp), DIMENSION(0:1) ::  ntdim_2d_xy  !< number of output intervals for 2d data (xy)
    INTEGER(iwp), DIMENSION(0:1) ::  ntdim_2d_xz  !< number of output intervals for 2d data (xz)
    INTEGER(iwp), DIMENSION(0:1) ::  ntdim_2d_yz  !< number of output intervals for 2d data (yz)
    INTEGER(iwp), DIMENSION(0:1) ::  ntdim_3d     !< number of output intervals for 3d data

    INTEGER(iwp), DIMENSION(max_masks,mask_xyz_dimension) ::  mask_k_over_surface = -1  !< namelist parameter, k index of height
                                                                                        !<over surface

    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  grid_level_count  !< internal switch for steering the multigrid v- and w-cycles

    INTEGER(iwp), DIMENSION(:,:), ALLOCATABLE ::  mask_i         !< subdomain grid index of masked output point on x-dimension
    INTEGER(iwp), DIMENSION(:,:), ALLOCATABLE ::  mask_j         !< subdomain grid index of masked output point on y-dimension
    INTEGER(iwp), DIMENSION(:,:), ALLOCATABLE ::  mask_k         !< subdomain grid index of masked output point on z-dimension
    INTEGER(iwp), DIMENSION(:,:), ALLOCATABLE ::  mask_i_global  !< global grid index of masked output point on x-dimension
    INTEGER(iwp), DIMENSION(:,:), ALLOCATABLE ::  mask_j_global  !< global grid index of masked output point on y-dimension
    INTEGER(iwp), DIMENSION(:,:), ALLOCATABLE ::  mask_k_global  !< global grid index of masked output point on z-dimension

    LOGICAL ::  advanced_div_correction = .FALSE.                !< namelist parameter
    LOGICAL ::  agent_time_unlimited = .FALSE.                   !< namelist parameter
    LOGICAL ::  air_chemistry = .FALSE.                          !< chemistry model switch
    LOGICAL ::  allow_negative_scalar_values = .FALSE.           !< namelist parameter
    LOGICAL ::  allow_roughness_limitation = .FALSE.             !< namelist parameter
    LOGICAL ::  atmosphere_run_coupled_to_ocean = .FALSE.        !< atmosphere part of a coupled atmosphere-ocean run
    LOGICAL ::  bc_dirichlet_l = .FALSE.                         !< flag indicating dirichlet boundary condition on left model
                                                                 !< boundary
    LOGICAL ::  bc_dirichlet_n = .FALSE.                         !< flag indicating dirichlet boundary condition on north model
                                                                 !< boundary
    LOGICAL ::  bc_dirichlet_r = .FALSE.                         !< flag indicating dirichlet boundary condition on right model
                                                                 !< boundary
    LOGICAL ::  bc_dirichlet_s = .FALSE.                         !< flag indicating dirichlet boundary condition on south model
                                                                 !< boundary
    LOGICAL ::  bc_lr_cyc =.TRUE.                                !< left-right boundary condition cyclic?
    LOGICAL ::  bc_lr_dirrad = .FALSE.                           !< left-right boundary condition dirichlet/radiation?
    LOGICAL ::  bc_lr_raddir = .FALSE.                           !< left-right boundary condition radiation/dirichlet?
    LOGICAL ::  bc_ns_cyc = .TRUE.                               !< north-south boundary condition cyclic?
    LOGICAL ::  bc_ns_dirrad = .FALSE.                           !< north-south boundary condition dirichlet/radiation?
    LOGICAL ::  bc_ns_raddir = .FALSE.                           !< north-south boundary condition radiation/dirichlet?
    LOGICAL ::  bc_radiation_l = .FALSE.                         !< radiation boundary condition for outflow at left domain boundary
    LOGICAL ::  bc_radiation_n = .FALSE.                         !< radiation boundary condition for outflow at north domain
                                                                 !< boundary
    LOGICAL ::  bc_radiation_r = .FALSE.                         !< radiation boundary condition for outflow at right domain
                                                                 !< boundary
    LOGICAL ::  bc_radiation_s = .FALSE.                         !< radiation boundary condition for outflow at south domain
                                                                 !< boundary
    LOGICAL ::  biometeorology = .FALSE.                         !< biometeorology module switch
    LOGICAL ::  calc_soil_moisture_during_spinup = .FALSE.       !< namelist parameter
    LOGICAL ::  call_psolver_at_all_substeps = .TRUE.            !< namelist parameter
    LOGICAL ::  check_realistic_q = .TRUE.                       !< namelist parameter
    LOGICAL ::  child_domain  = .FALSE.                          !< flag indicating that model is nested in a parent domain
    LOGICAL ::  cloud_droplets = .FALSE.                         !< namelist parameter
    LOGICAL ::  conserve_volume_flow = .FALSE.                   !< namelist parameter
    LOGICAL ::  constant_diffusion = .FALSE.                     !< diffusion coefficient constant?
    LOGICAL ::  constant_flux_layer = .TRUE.                     !< namelist parameter
    LOGICAL ::  constant_heatflux = .TRUE.                       !< heat flux at all surfaces constant?
    LOGICAL ::  constant_top_heatflux = .TRUE.                   !< heat flux at domain top constant?
    LOGICAL ::  constant_top_momentumflux = .FALSE.              !< momentum flux at domain topconstant?
    LOGICAL ::  constant_top_salinityflux = .TRUE.               !< constant salinity flux at ocean surface
    LOGICAL ::  constant_top_scalarflux = .TRUE.                 !< passive-scalar flux at domain top constant?
    LOGICAL ::  constant_scalarflux = .TRUE.                     !< passive-scalar flux at surfaces constant?
    LOGICAL ::  constant_waterflux = .TRUE.                      !< water flux at all surfaces constant?
    LOGICAL ::  create_disturbances = .TRUE.                     !< namelist parameter
    LOGICAL ::  cyclic_fill_initialization = .FALSE.             !< switch for steering cyclic fill actions
    LOGICAL ::  data_output_during_spinup = .FALSE.              !< namelist parameter
    LOGICAL ::  data_output_2d_on_each_pe = .TRUE.               !< namelist parameter
    LOGICAL ::  dcep = .FALSE.                                   !< switch for activiating the dcep model
    LOGICAL ::  debug_output = .FALSE.                           !< namelist parameter
    LOGICAL ::  debug_output_timestep = .FALSE.                  !< namelist parameter
    LOGICAL ::  disturbance_created = .FALSE.                    !< flow disturbance imposed?
    LOGICAL ::  do2d_at_begin = .FALSE.                          !< namelist parameter
    LOGICAL ::  do3d_at_begin = .FALSE.                          !< namelist parameter
    LOGICAL ::  do_sum = .FALSE.                                 !< contribute to time average of profile data?
    LOGICAL ::  dp_external = .FALSE.                            !< namelist parameter
    LOGICAL ::  dp_smooth = .FALSE.                              !< namelist parameter
    LOGICAL ::  dt_fixed = .FALSE.                               !< fixed timestep (namelist parameter dt set)?
    LOGICAL ::  dt_3d_reached                                    !< internal timestep for particle advection
    LOGICAL ::  dt_3d_reached_l                                  !< internal timestep for particle advection
    LOGICAL ::  external_anthropogenic_heat = .FALSE.            !< switch for anthropogenic heat from external building/transport model or other sources
    LOGICAL ::  first_call_mas = .TRUE.                          !< call mas only once per timestep ??
    LOGICAL ::  force_print_header = .FALSE.                     !< namelist parameter
    LOGICAL ::  galilei_transformation = .FALSE.                 !< namelist parameter
    LOGICAL ::  homogenize_surface_temperature = .FALSE.         !< namelist parameter
    LOGICAL ::  humidity = .FALSE.                               !< namelist parameter
    LOGICAL ::  humidity_remote = .FALSE.                        !< switch for receiving near-surface humidity flux
                                                                 !< (atmosphere-ocean coupling)
    LOGICAL ::  indoor_model = .FALSE.                           !< switch for indoor-climate and energy-demand model
    LOGICAL ::  interpolate_to_grid_center = .FALSE.             !< namelist parameter
    LOGICAL ::  kolmogorov_length_scale = .FALSE.                !< switch to activate calculations in flow_statistics for the
                                                                 !< kolmogorov length scale
    LOGICAL ::  large_scale_forcing = .FALSE.                    !< namelist parameter
    LOGICAL ::  large_scale_subsidence = .FALSE.                 !< namelist parameter
    LOGICAL ::  land_surface = .FALSE.                           !< use land surface model?
    LOGICAL ::  les_dai = .FALSE.                                !< use Dai et al. turbulence closure (modified 1.5-order closure)
                                                                 !< for LES mode. Shall replace the default 1.5-order closure
    LOGICAL ::  les_dynamic = .FALSE.                            !< use dynamic subgrid model as turbulence closure for LES mode
    LOGICAL ::  les_default = .FALSE.                            !< use 1.5-order default turbulence closure for LES mode
    LOGICAL ::  lsf_exception = .FALSE.                          !< use of lsf with buildings (temporary)?
    LOGICAL ::  lsf_surf = .TRUE.                                !< use surface forcing (large scale forcing)?
    LOGICAL ::  lsf_vert = .TRUE.                                !< use atmospheric forcing (large scale forcing)?
    LOGICAL ::  masking_method = .FALSE.                         !< namelist parameter
    LOGICAL ::  mg_switch_to_pe0 = .FALSE.                       !< internal multigrid switch for steering the ghost point exchange
                                                                 !< in case that data has been collected on PE0
    LOGICAL ::  monotonic_limiter_z = .FALSE.                    !< use monotonic flux limiter for vertical scalar advection
    LOGICAL ::  nested_run = .FALSE.                             !< general switch to indicate a nested run
    LOGICAL ::  nesting_offline = .FALSE.                        !< flag controlling offline nesting in COSMO model
    LOGICAL ::  neutral = .FALSE.                                !< namelist parameter
    LOGICAL ::  nudging = .FALSE.                                !< namelist parameter
    LOGICAL ::  ocean_mode = .FALSE.                             !< namelist parameter
    LOGICAL ::  ocean_run_coupled_to_atmosphere = .FALSE.        !< ocean part of coupled atmosphere-ocean run
    LOGICAL ::  open_debug_files = .TRUE.                        !< switch for opening the debug files independent of switch debug_output
    LOGICAL ::  passive_scalar = .FALSE.                         !< namelist parameter
    LOGICAL ::  pe_grid_prescribed = .FALSE.                     !< switch to indicate if PE grid is prescribed by user
    LOGICAL ::  plant_canopy = .FALSE.                           !< switch for use of plant canopy model
    LOGICAL ::  random_heatflux = .FALSE.                        !< namelist parameter
    LOGICAL ::  rans_mode = .FALSE.                              !< switch between RANS and LES mode
    LOGICAL ::  rans_tke_e = .FALSE.                             !< use TKE-e turbulence closure for RANS mode
    LOGICAL ::  rans_tke_l = .FALSE.                             !< use TKE-l turbulence closure for RANS mode
    LOGICAL ::  read_spinup_data = .FALSE.                       !< flag to control the input of surface spinup data
    LOGICAL ::  read_svf = .FALSE.                               !< ENVPAR namelist parameter to steer input of svf
                                                                 !< (ENVPAR is provided by palmrun)
    LOGICAL ::  run_control_header = .FALSE.                     !< onetime output of RUN_CONTROL header
    LOGICAL ::  salinity = .TRUE.                                !< switch for using salinity
    LOGICAL ::  salsa = .FALSE.                                  !< switch for the sectional aerosol module salsa
    LOGICAL ::  scalar_rayleigh_damping = .TRUE.                 !< namelist parameter
    LOGICAL ::  serial_run = .FALSE.                             !< switch to indicate serial or parallel run
    LOGICAL ::  sloping_surface = .FALSE.                        !< use sloped surface? (namelist parameter alpha_surface)
    LOGICAL ::  spinup = .FALSE.                                 !< perform model spinup without atmosphere code?
    LOGICAL ::  surface_output = .FALSE.                         !< output of surface data
    LOGICAL ::  stop_dt = .FALSE.                                !< internal switch to stop the time stepping
    LOGICAL ::  synchronous_exchange = .FALSE.                   !< namelist parameter
    LOGICAL ::  syn_turb_gen = .FALSE.                           !< flag for synthetic turbulence generator module
    LOGICAL ::  temperton_fft_vec = .FALSE.                      !< flag for using vectorized version of Temperton FFT
    LOGICAL ::  terminate_run = .FALSE.                          !< terminate run (cpu-time limit, restarts)?
    LOGICAL ::  terrain_following_mapping = .FALSE.              !< namelist parameter
    LOGICAL ::  topo_no_distinct = .FALSE.                       !< flag controlling classification of topography surfaces
    LOGICAL ::  turbulent_inflow = .FALSE.                       !< namelist parameter
    LOGICAL ::  turbulent_outflow = .FALSE.                      !< namelist parameter
    LOGICAL ::  urban_surface = .FALSE.                          !< use urban surface model?
    LOGICAL ::  use_fixed_date = .FALSE.                         !< date of simulation does not change (namelist parameter)
    LOGICAL ::  use_fixed_time = .FALSE.                         !< time of simulation does not change (namelist parameter)
    LOGICAL ::  use_free_convection_scaling = .FALSE.            !< namelist parameter to switch on free convection velocity scale
                                                                 !< in calculation of horizontal wind speed (surface_layer_fluxes)
    LOGICAL ::  use_initial_profile_as_reference = .FALSE.       !< use of initial profiles as reference state?
    LOGICAL ::  use_prescribed_profile_data = .FALSE.            !< use of prescribed wind profiles?
                                                                 !< (namelist parameters u_profile, v_profile)
    LOGICAL ::  use_single_reference_value = .FALSE.             !< use of single value as reference state?
    LOGICAL ::  use_sm_for_poisfft = .FALSE.                     !< use shared-memory on node for solving the Poisson equation with FFT-method (1d-decomposition)
    LOGICAL ::  use_subsidence_tendencies = .FALSE.              !< namelist parameter
    LOGICAL ::  use_surface_fluxes = .FALSE.                     !< namelist parameter
    LOGICAL ::  use_top_fluxes = .FALSE.                         !< namelist parameter
    LOGICAL ::  use_ug_for_galilei_tr = .TRUE.                   !< namelist parameter
    LOGICAL ::  use_upstream_for_tke = .FALSE.                   !< namelist parameter
    LOGICAL ::  vdi_checks = .FALSE.                             !< do internal controls after VDI 3783 Part 9
    LOGICAL ::  virtual_flight = .FALSE.                         !< use virtual flight model
    LOGICAL ::  virtual_measurement = .FALSE.                    !< control parameter to switch-on virtual measurements
    LOGICAL ::  wall_adjustment = .TRUE.                         !< namelist parameter
    LOGICAL ::  wind_turbine = .FALSE.                           !< flag for use of wind turbine model
    LOGICAL ::  write_binary = .FALSE.                           !< ENVPAR namelist parameter to steer restart I/O
                                                                 !< (ENVPAR is provided by palmrun)
    LOGICAL ::  write_spinup_data = .FALSE.                      !< ENVPAR namelist parameter to steer restart I/O
                                                                 !< (ENVPAR is provided by palmrun)
    LOGICAL ::  write_svf = .FALSE.                              !< ENVPAR namelist parameter to steer output of svf
                                                                 !< (ENVPAR is provided by palmrun)
    LOGICAL ::  ws_scheme_sca = .FALSE.                          !< use Wicker-Skamarock scheme (scalar advection)?
    LOGICAL ::  ws_scheme_mom = .FALSE.                          !< use Wicker-Skamarock scheme (momentum advection)?

    LOGICAL ::  data_output_xy(0:1) = .FALSE.                !< output of xy cross-section data?
    LOGICAL ::  data_output_xz(0:1) = .FALSE.                !< output of xz cross-section data?
    LOGICAL ::  data_output_yz(0:1) = .FALSE.                !< output of yz cross-section data?

    LOGICAL, DIMENSION(max_masks) ::  mask_surface = .FALSE.   !< flag for surface-following masked output

    REAL(wp) ::  advected_distance_x = 0.0_wp                  !< advected distance of model domain along x
                                                               !< (galilei transformation)
    REAL(wp) ::  advected_distance_y = 0.0_wp                  !< advected distance of model domain along y
                                                               !< (galilei transformation)
    REAL(wp) ::  alpha_surface = 0.0_wp                        !< namelist parameter
    REAL(wp) ::  atmos_ocean_sign = 1.0_wp                     !< vertical-grid conversion factor
                                                               !< (=1.0 in atmosphere, =-1.0 in ocean)
    REAL(wp) ::  averaging_interval = 0.0_wp                   !< namelist parameter
    REAL(wp) ::  averaging_interval_pr = 9999999.9_wp          !< namelist parameter
    REAL(wp) ::  bc_pt_t_val                                   !< vertical gradient of pt near domain top
    REAL(wp) ::  bc_q_t_val                                    !< vertical gradient of humidity near domain top
    REAL(wp) ::  bc_s_t_val                                    !< vertical gradient of passive scalar near domain top
    REAL(wp) ::  bottom_salinityflux = 0.0_wp                  !< namelist parameter
    REAL(wp) ::  building_height = 50.0_wp                     !< namelist parameter
    REAL(wp) ::  building_length_x = 50.0_wp                   !< namelist parameter
    REAL(wp) ::  building_length_y = 50.0_wp                   !< namelist parameter
    REAL(wp) ::  building_wall_left = 9999999.9_wp             !< namelist parameter
    REAL(wp) ::  building_wall_south = 9999999.9_wp            !< namelist parameter
    REAL(wp) ::  canyon_height = 50.0_wp                       !< namelist parameter
    REAL(wp) ::  canyon_width_x = 9999999.9_wp                 !< namelist parameter
    REAL(wp) ::  canyon_width_y = 9999999.9_wp                 !< namelist parameter
    REAL(wp) ::  canyon_wall_left = 9999999.9_wp               !< namelist parameter
    REAL(wp) ::  canyon_wall_south = 9999999.9_wp              !< namelist parameter
    REAL(wp) ::  cfl_factor = -1.0_wp                          !< namelist parameter
    REAL(wp) ::  cos_alpha_surface                             !< cosine of alpha_surface
    REAL(wp) ::  coupling_start_time = 0.0_wp                  !< namelist parameter
    REAL(wp) ::  days_since_reference_point = 0.0_wp           !< days after atmosphere-ocean coupling has been activated,
                                                               !< or after spinup phase of LSM has been finished
    REAL(wp) ::  disturbance_amplitude = 0.25_wp               !< namelist parameter
    REAL(wp) ::  disturbance_energy_limit = 0.01_wp            !< namelist parameter
    REAL(wp) ::  disturbance_level_b = -9999999.9_wp           !< namelist parameter
    REAL(wp) ::  disturbance_level_t = -9999999.9_wp           !< namelist parameter
    REAL(wp) ::  dp_level_b = 0.0_wp                           !< namelist parameter
    REAL(wp) ::  dt = -1.0_wp                                  !< namelist parameter
    REAL(wp) ::  dt_averaging_input = 0.0_wp                   !< namelist parameter
    REAL(wp) ::  dt_averaging_input_pr = 9999999.9_wp          !< namelist parameter
    REAL(wp) ::  dt_coupling = 9999999.9_wp                    !< namelist parameter
    REAL(wp) ::  dt_data_output = 9999999.9_wp                 !< namelist parameter
    REAL(wp) ::  dt_data_output_av = 9999999.9_wp              !< namelist parameter
    REAL(wp) ::  dt_disturb = 9999999.9_wp                     !< namelist parameter
    REAL(wp) ::  dt_dopr = 9999999.9_wp                        !< namelist parameter
    REAL(wp) ::  dt_dopr_listing = 9999999.9_wp                !< namelist parameter
    REAL(wp) ::  dt_dots = 9999999.9_wp                        !< namelist parameter
    REAL(wp) ::  dt_do2d_xy = 9999999.9_wp                     !< namelist parameter
    REAL(wp) ::  dt_do2d_xz = 9999999.9_wp                     !< namelist parameter
    REAL(wp) ::  dt_do2d_yz = 9999999.9_wp                     !< namelist parameter
    REAL(wp) ::  dt_do3d = 9999999.9_wp                        !< namelist parameter
    REAL(wp) ::  dt_max = 20.0_wp                              !< namelist parameter
    REAL(wp) ::  dt_restart = 9999999.9_wp                     !< namelist parameter
    REAL(wp) ::  dt_run_control = 60.0_wp                      !< namelist parameter
    REAL(wp) ::  dt_spinup = 5.0_wp                            !< namelist parameter
    REAL(wp) ::  dt_write_agent_data = 9999999.9_wp            !< namelist parameter
    REAL(wp) ::  dt_3d = 0.01_wp                               !< time step
    REAL(wp) ::  dz_max = 999.0_wp                             !< namelist parameter
    REAL(wp) ::  dz_stretch_factor = 1.08_wp                   !< namelist parameter
    REAL(wp) ::  dz_stretch_level = -9999999.9_wp              !< namelist parameter
    REAL(wp) ::  e_init = 0.0_wp                               !< namelist parameter
    REAL(wp) ::  e_min = 0.0_wp                                !< namelist parameter
    REAL(wp) ::  end_time = 0.0_wp                             !< namelist parameter
    REAL(wp) ::  f = 0.0_wp                                    !< Coriolis parameter
    REAL(wp) ::  fs = 0.0_wp                                   !< Coriolis parameter
    REAL(wp) ::  km_constant = -1.0_wp                         !< namelist parameter
    REAL(wp) ::  latitude = 55.0_wp                            !< namelist parameter
    REAL(wp) ::  longitude = 0.0_wp                            !< namelist parameter
    REAL(wp) ::  mask_scale_x = 1.0_wp                         !< namelist parameter
    REAL(wp) ::  mask_scale_y = 1.0_wp                         !< namelist parameter
    REAL(wp) ::  mask_scale_z = 1.0_wp                         !< namelist parameter
    REAL(wp) ::  maximum_cpu_time_allowed = 0.0_wp             !< given wall time for run
    REAL(wp) ::  molecular_viscosity = 1.461E-5_wp             !< molecular viscosity (used in lsm and lpm)
    REAL(wp) ::  multi_agent_system_end   = 9999999.9_wp       !< namelist parameter (see documentation)
    REAL(wp) ::  multi_agent_system_start = 0.0_wp             !< namelist parameter (see documentation)
    REAL(wp) ::  omega = 7.29212E-5_wp                         !< namelist parameter
    REAL(wp) ::  omega_sor = 1.8_wp                            !< namelist parameter
    REAL(wp) ::  outflow_damping_factor = 0.0_wp               !< namelist parameter
    REAL(wp) ::  outflow_damping_width = 0.0_wp                !< namelist parameter
    REAL(wp) ::  outflow_source_plane = -9999999.9_wp          !< namelist parameter
    REAL(wp) ::  output_3d_file_size = 0.0_wp                  !< file size of 3d NetCDF output file
    REAL(wp) ::  particle_maximum_age = 9999999.9_wp           !< namelist parameter
    REAL(wp) ::  prandtl_number = 1.0_wp                       !< namelist parameter
    REAL(wp) ::  pt_damping_factor = 0.0_wp                    !< namelist parameter
    REAL(wp) ::  pt_damping_width = 0.0_wp                     !< namelist parameter
    REAL(wp) ::  pt_reference = 9999999.9_wp                   !< namelist parameter
    REAL(wp) ::  pt_slope_offset = 0.0_wp                      !< temperature difference between left and right
                                                               !< boundary of total domain
    REAL(wp) ::  pt_surface = 300.0_wp                         !< namelist parameter
    REAL(wp) ::  pt_surface_heating_rate = 0.0_wp              !< namelist parameter
    REAL(wp) ::  pt_surface_initial_change = 0.0_wp            !< namelist parameter
    REAL(wp) ::  q_surface = 0.0_wp                            !< namelist parameter
    REAL(wp) ::  q_surface_initial_change = 0.0_wp             !< namelist parameter
    REAL(wp) ::  rayleigh_damping_factor = 0.0_wp              !< namelist parameter
    REAL(wp) ::  rayleigh_damping_height = -1.0_wp             !< namelist parameter
    REAL(wp) ::  residual_limit = 1.0E-4_wp                    !< namelist parameter
    REAL(wp) ::  restart_file_size                             !< size of restart file in mbyte
    REAL(wp) ::  restart_time = 9999999.9_wp                   !< namelist parameter
    REAL(wp) ::  rho_reference                                 !< reference state of density
    REAL(wp) ::  rho_surface                                   !< surface value of density
    REAL(wp) ::  rotation_angle = 0.0_wp                       !< clockwise rotation of model North relative to real North [deg]
    REAL(wp) ::  roughness_length = 0.1_wp                     !< namelist parameter
    REAL(wp) ::  simulated_time = 0.0_wp                       !< elapsed simulated time
    REAL(wp) ::  simulated_time_at_begin                       !< elapsed simulated time of previous run (job chain)
    REAL(wp) ::  sin_alpha_surface                             !< sine of alpha_surface (sloped surface)
    REAL(wp) ::  skip_time_data_output = 0.0_wp                !< namelist parameter
    REAL(wp) ::  skip_time_data_output_av = 9999999.9_wp       !< namelist parameter
    REAL(wp) ::  skip_time_dopr = 9999999.9_wp                 !< namelist parameter
    REAL(wp) ::  skip_time_do2d_xy = 9999999.9_wp              !< namelist parameter
    REAL(wp) ::  skip_time_do2d_xz = 9999999.9_wp              !< namelist parameter
    REAL(wp) ::  skip_time_do2d_yz = 9999999.9_wp              !< namelist parameter
    REAL(wp) ::  skip_time_do3d = 9999999.9_wp                 !< namelist parameter
    REAL(wp) ::  spinup_pt_amplitude = 0.0_wp                  !< namelist parameter
    REAL(wp) ::  spinup_pt_mean = 9999999.9_wp                 !< namelist parameter
    REAL(wp) ::  spinup_time = 0.0_wp                          !< namelist parameter
    REAL(wp) ::  surface_heatflux = 9999999.9_wp               !< namelist parameter
    REAL(wp) ::  surface_pressure = 1013.25_wp                 !< namelist parameter
    REAL(wp) ::  surface_scalarflux = 9999999.9_wp             !< namelist parameter
    REAL(wp) ::  surface_waterflux = 9999999.9_wp              !< namelist parameter
    REAL(wp) ::  s_surface = 0.0_wp                            !< namelist parameter
    REAL(wp) ::  s_surface_initial_change = 0.0_wp             !< namelist parameter
    REAL(wp) ::  termination_time_needed = 35.0_wp             !< namelist parameter
    REAL(wp) ::  time_coupling = 0.0_wp                        !< time since last coupling (surface_coupler)
    REAL(wp) ::  time_disturb = 0.0_wp                         !< time since last flow disturbance
    REAL(wp) ::  time_dopr = 0.0_wp                            !< time since last profile output
    REAL(wp) ::  time_dopr_av = 0.0_wp                         !< time since last averaged profile output
    REAL(wp) ::  time_dopr_listing = 0.0_wp                    !< time since last profile output (ASCII) on file
    REAL(wp) ::  time_dosp = 0.0_wp                            !< time since last spectra output
    REAL(wp) ::  time_dosp_av = 0.0_wp                         !< time since last averaged spectra output
    REAL(wp) ::  time_dots = 0.0_wp                            !< time since last timeseries output
    REAL(wp) ::  time_do2d_xy = 0.0_wp                         !< time since last xy cross-section output
    REAL(wp) ::  time_do2d_xz = 0.0_wp                         !< time since last xz cross-section output
    REAL(wp) ::  time_do2d_yz = 0.0_wp                         !< time since last yz cross-section output
    REAL(wp) ::  time_do3d = 0.0_wp                            !< time since last 3d output
    REAL(wp) ::  time_do_av = 0.0_wp                           !< time since last averaged-data output
    REAL(wp) ::  time_do_sla = 0.0_wp                          !< time since last
    REAL(wp) ::  time_restart = 9999999.9_wp                   !< time at which run shall be terminated and restarted
    REAL(wp) ::  time_run_control = 0.0_wp                     !< time since last RUN_CONTROL output
    REAL(wp) ::  time_since_reference_point = 0.0_wp           !< time after atmosphere-ocean coupling has been activated, or time
                                                               !< after spinup phase of LSM has been finished
    REAL(wp) ::  top_heatflux = 9999999.9_wp                   !< namelist parameter
    REAL(wp) ::  top_momentumflux_u = 9999999.9_wp             !< namelist parameter
    REAL(wp) ::  top_momentumflux_v = 9999999.9_wp             !< namelist parameter
    REAL(wp) ::  top_salinityflux = 9999999.9_wp               !< namelist parameter
    REAL(wp) ::  top_scalarflux = 9999999.9_wp                 !< namelist parameter
    REAL(wp) ::  tunnel_height = 9999999.9_wp                  !< namelist parameter
    REAL(wp) ::  tunnel_length = 9999999.9_wp                  !< namelist parameter
    REAL(wp) ::  tunnel_width_x = 9999999.9_wp                 !< namelist parameter
    REAL(wp) ::  tunnel_width_y = 9999999.9_wp                 !< namelist parameter
    REAL(wp) ::  tunnel_wall_depth = 9999999.9_wp              !< namelist parameter
    REAL(wp) ::  ug_surface = 0.0_wp                           !< namelist parameter
    REAL(wp) ::  u_bulk = 0.0_wp                               !< namelist parameter
    REAL(wp) ::  u_gtrans = 0.0_wp                             !< transformed wind component (galilei transformation)
    REAL(wp) ::  vg_surface = 0.0_wp                           !< namelist parameter
    REAL(wp) ::  vpt_reference = 9999999.9_wp                  !< reference state of virtual potential temperature
    REAL(wp) ::  v_bulk = 0.0_wp                               !< namelist parameter
    REAL(wp) ::  v_gtrans = 0.0_wp                             !< transformed wind component (galilei transformation)
    REAL(wp) ::  zeta_max = 20.0_wp                            !< namelist parameter
    REAL(wp) ::  zeta_min = -20.0_wp                           !< namelist parameter
    REAL(wp) ::  z0h_factor = 1.0_wp                           !< namelist parameter

    REAL(wp) ::  do2d_xy_last_time(0:1) = -1.0_wp                  !< time of previous xy output
    REAL(wp) ::  do2d_xz_last_time(0:1) = -1.0_wp                  !< time of previous xz output
    REAL(wp) ::  do2d_yz_last_time(0:1) = -1.0_wp                  !< time of previous yz output
    REAL(wp) ::  dpdxy(1:2) = 0.0_wp                               !< namelist parameter
    REAL(wp) ::  dt_domask(max_masks) = 9999999.9_wp               !< namelist parameter
    REAL(wp) ::  dz(10) = -1.0_wp                                  !< namelist parameter
    REAL(wp) ::  dz_stretch_level_start(9) = -9999999.9_wp         !< namelist parameter
    REAL(wp) ::  dz_stretch_level_end(9) = 9999999.9_wp            !< namelist parameter
    REAL(wp) ::  dz_stretch_factor_array(9) = 1.08_wp              !< namelist parameter
    REAL(wp) ::  mask_scale(3)                                     !< collective array for mask_scale_x/y/z
    REAL(wp) ::  pt_vertical_gradient(10) = 0.0_wp                 !< namelist parameter
    REAL(wp) ::  pt_vertical_gradient_level(10) = -999999.9_wp     !< namelist parameter
    REAL(wp) ::  q_vertical_gradient(10) = 0.0_wp                  !< namelist parameter
    REAL(wp) ::  q_vertical_gradient_level(10) = -999999.9_wp      !< namelist parameter
    REAL(wp) ::  section_xy_m(100) = -9999999.9_wp                 !< namelist parameter
    REAL(wp) ::  section_xz_m(100) = -9999999.9_wp                 !< namelist parameter
    REAL(wp) ::  section_yz_m(100) = -9999999.9_wp                 !< namelist parameter
    REAL(wp) ::  s_vertical_gradient(10) = 0.0_wp                  !< namelist parameter
    REAL(wp) ::  s_vertical_gradient_level(10) = -999999.9_wp      !< namelist parameter
    REAL(wp) ::  skip_time_domask(max_masks) = 9999999.9_wp        !< namelist parameter
    REAL(wp) ::  threshold(20) = 0.0_wp                            !< namelist parameter
    REAL(wp) ::  time_domask(max_masks) = 0.0_wp                   !< namelist parameter
    REAL(wp) ::  tsc(10) = (/ 1.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, &    !< array used for controlling time-integration at different substeps
                 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp /)
    REAL(wp) ::  u_profile(200) = 9999999.9_wp                     !< namelist parameter
    REAL(wp) ::  uv_heights(200) = 9999999.9_wp                    !< namelist parameter
    REAL(wp) ::  v_profile(200) = 9999999.9_wp                     !< namelist parameter
    REAL(wp) ::  ug_vertical_gradient(10) = 0.0_wp                 !< namelist parameter
    REAL(wp) ::  ug_vertical_gradient_level(10) = -9999999.9_wp    !< namelist parameter
    REAL(wp) ::  vg_vertical_gradient(10) = 0.0_wp                 !< namelist parameter
    REAL(wp) ::  vg_vertical_gradient_level(10) = -9999999.9_wp    !< namelist parameter
    REAL(wp) ::  volume_flow(1:3) = 0.0_wp                         !< volume flow through 1:yz-plane, 2: xz-plane, 3: xy-plane
                                                                   !< (nest childs only)
    REAL(wp) ::  volume_flow_area(1:3) = 0.0_wp                    !< area of the respective volume flow planes
    REAL(wp) ::  volume_flow_initial(1:3) = 0.0_wp                 !< initial volume flow (t=0) through the respective volume flow
                                                                   !< planes
    REAL(wp) ::  wall_heatflux(0:5) = 0.0_wp                       !< namelist parameter
    REAL(wp) ::  wall_humidityflux(0:5) = 0.0_wp                   !< namelist parameter
    REAL(wp) ::  wall_scalarflux(0:5) = 0.0_wp                     !< namelist parameter
    REAL(wp) ::  subs_vertical_gradient(10) = 0.0_wp               !< namelist parameter
    REAL(wp) ::  subs_vertical_gradient_level(10) = -9999999.9_wp  !< namelist parameter

    REAL(wp), DIMENSION(:), ALLOCATABLE ::  dp_smooth_factor  !< smoothing factor for external pressure gradient forcing

    REAL(wp), DIMENSION(max_masks,mask_xyz_dimension) ::  mask_x = -1.0_wp  !< namelist parameter
    REAL(wp), DIMENSION(max_masks,mask_xyz_dimension) ::  mask_y = -1.0_wp  !< namelist parameter
    REAL(wp), DIMENSION(max_masks,mask_xyz_dimension) ::  mask_z = -1.0_wp  !< namelist parameter

    REAL(wp), DIMENSION(max_masks,3) ::  mask_x_loop = -1.0_wp  !< namelist parameter
    REAL(wp), DIMENSION(max_masks,3) ::  mask_y_loop = -1.0_wp  !< namelist parameter
    REAL(wp), DIMENSION(max_masks,3) ::  mask_z_loop = -1.0_wp  !< namelist parameter

!
!-- Internal mask arrays ("mask,dimension,selection")
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  mask       !< collective array for mask_x/y/z
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  mask_loop  !< collective array for mask_x/y/z_loop

    SAVE

 END MODULE control_parameters


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Definition of grid spacings.
!--------------------------------------------------------------------------------------------------!
 MODULE grid_variables

    USE kinds

    REAL(wp) ::  ddx          !< 1/dx
    REAL(wp) ::  ddx2         !< 1/dx2
    REAL(wp) ::  dx = 1.0_wp  !< horizontal grid size (along x-direction)
    REAL(wp) ::  dx2          !< dx*dx
    REAL(wp) ::  ddy          !< 1/dy
    REAL(wp) ::  ddy2         !< 1/dy2
    REAL(wp) ::  dy = 1.0_wp  !< horizontal grid size (along y-direction)
    REAL(wp) ::  dy2          !< dy*dy

    REAL(wp), DIMENSION(:), ALLOCATABLE ::  ddx2_mg  !< 1/dx_l**2 (dx_l: grid spacing along x on different multigrid level)
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  ddy2_mg  !< 1/dy_l**2 (dy_l: grid spacing along y on different multigrid level)

    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  zu_s_inner  !< height of topography top on scalar grid
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  zw_w_inner  !< height of topography top on w grid

    SAVE

 END MODULE grid_variables


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Definition of array bounds, number of gridpoints, and wall flag arrays.
!--------------------------------------------------------------------------------------------------!
 MODULE indices

    USE kinds

    INTEGER(iwp) ::  nbgp = 3       !< number of boundary ghost points
    INTEGER(iwp) ::  ngp_sums       !< number of vertical profile grid points time number of output profiles - used for allreduce
                                    !< statements in MPI calls
    INTEGER(iwp) ::  ngp_sums_ls    !< number of vertical profile grid points time number of large-scale forcing profiles - used for
                                    !< allreduce statements in MPI calls
    INTEGER(iwp) ::  nnx            !< number of subdomain grid points in x-direction
    INTEGER(iwp) ::  nx = 0         !< nx+1 = total number of grid points in x-direction
    INTEGER(iwp) ::  nxl            !< left-most grid index of subdomain (excluding ghost points)
    INTEGER(iwp) ::  nxlg           !< left-most grid index of subdomain (including ghost points)
    INTEGER(iwp) ::  nxlu           !< =nxl+1 (at left domain boundary with inflow from left), else =nxl
                                    !< (used for u-velocity component)
    INTEGER(iwp) ::  nxr            !< right-most grid index of subdomain (excluding ghost points)
    INTEGER(iwp) ::  nxrg           !< right-most grid index of subdomain (including ghost points)
    INTEGER(iwp) ::  nx_on_file     !< nx of previous run in job chain
    INTEGER(iwp) ::  nny            !< number of subdomain grid points in y-direction
    INTEGER(iwp) ::  ny = 0         !< ny+1 = total number of grid points in y-direction
    INTEGER(iwp) ::  nyn            !< north-most grid index of subdomain (excluding ghost points)
    INTEGER(iwp) ::  nyng           !< north-most grid index of subdomain (including ghost points)
    INTEGER(iwp) ::  nys            !< south-most grid index of subdomain (excluding ghost points)
    INTEGER(iwp) ::  nysg           !< south-most grid index of subdomain (including ghost points)
    INTEGER(iwp) ::  nysv           !< =nys+1 (at south domain boundary with inflow from south), else =nys
                                    !< (used for v-velocity component)
    INTEGER(iwp) ::  ny_on_file     !< ny of previous run in job chain
    INTEGER(iwp) ::  nnz            !< number of subdomain grid points in z-direction
    INTEGER(iwp) ::  nz = 0         !< total number of grid points in z-direction
    INTEGER(iwp) ::  nzb            !< bottom grid index of computational domain
    INTEGER(iwp) ::  nzb_diff       !< will be removed
    INTEGER(iwp) ::  nzb_max        !< vertical index of topography top
    INTEGER(iwp) ::  nzt            !< nzt+1 = top grid index of computational domain
    INTEGER(iwp) ::  topo_min_level !< minimum topography-top index (usually equal to nzb)

    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  nnx_pe        !< grid points along x-direction for every PE
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  nny_pe        !< grid points along y-direction for every PE
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  nxl_pe        !< lower index bound along x-direction for every PE
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  nxr_pe        !< upper index bound along x-direction for every PE
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  nyn_pe        !< lower index bound along y-direction for every PE
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  nys_pe        !< lower index bound along y-direction for every PE

    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  ngp_2dh       !< number of grid points of a horizontal cross section through the
                                                              !< total domain
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  ngp_2dh_wgrid !< number of prognostic w-grid points of a horizontal cross section
                                                              !< through the total domain
    INTEGER(idp), DIMENSION(:), ALLOCATABLE ::  ngp_3d        !< number of grid points of the total domain
    INTEGER(idp), DIMENSION(:), ALLOCATABLE ::  ngp_3d_inner  !< ! need to have 64 bit for grids > 2E9
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  nxl_mg        !< left-most grid index of subdomain on different multigrid level
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  nxr_mg        !< right-most grid index of subdomain on different multigrid level
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  nyn_mg        !< north-most grid index of subdomain on different multigrid level
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  nys_mg        !< south-most grid index of subdomain on different multigrid level
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  nzt_mg        !< top-most grid index of subdomain on different multigrid level


    INTEGER(iwp), DIMENSION(:,:), ALLOCATABLE ::  mg_loc_ind        !< internal array to store index bounds of all PEs of that
                                                                    !< multigrid level where data is collected to PE0
    INTEGER(iwp), DIMENSION(:,:), ALLOCATABLE ::  ngp_2dh_outer     !< number of horizontal grid points which are non-topography and
                                                                    !< non-surface-bounded
    INTEGER(iwp), DIMENSION(:,:), ALLOCATABLE ::  ngp_2dh_s_inner   !< number of horizontal grid points which are non-topography

    INTEGER(iwp), DIMENSION(:,:,:), ALLOCATABLE,  TARGET ::  topo_flags_1   !< topograpyh masking flag on multigrid level 1
    INTEGER(iwp), DIMENSION(:,:,:), ALLOCATABLE,  TARGET ::  topo_flags_2   !< topograpyh masking flag on multigrid level 2
    INTEGER(iwp), DIMENSION(:,:,:), ALLOCATABLE,  TARGET ::  topo_flags_3   !< topograpyh masking flag on multigrid level 3
    INTEGER(iwp), DIMENSION(:,:,:), ALLOCATABLE,  TARGET ::  topo_flags_4   !< topograpyh masking flag on multigrid level 4
    INTEGER(iwp), DIMENSION(:,:,:), ALLOCATABLE,  TARGET ::  topo_flags_5   !< topograpyh masking flag on multigrid level 5
    INTEGER(iwp), DIMENSION(:,:,:), ALLOCATABLE,  TARGET ::  topo_flags_6   !< topograpyh masking flag on multigrid level 6
    INTEGER(iwp), DIMENSION(:,:,:), ALLOCATABLE,  TARGET ::  topo_flags_7   !< topograpyh masking flag on multigrid level 7
    INTEGER(iwp), DIMENSION(:,:,:), ALLOCATABLE,  TARGET ::  topo_flags_8   !< topograpyh masking flag on multigrid level 8
    INTEGER(iwp), DIMENSION(:,:,:), ALLOCATABLE,  TARGET ::  topo_flags_9   !< topograpyh masking flag on multigrid level 9
    INTEGER(iwp), DIMENSION(:,:,:), ALLOCATABLE,  TARGET ::  topo_flags_10  !< topograpyh masking flag on multigrid level 10

    INTEGER(iwp), DIMENSION(:,:,:), ALLOCATABLE ::  advc_flags_m            !< flags used to degrade order of advection scheme for
                                                                            !< momentum
    INTEGER(iwp), DIMENSION(:,:,:), ALLOCATABLE ::  advc_flags_s            !< flags used to degrade order of advection scheme for
                                                                            !< scalar quantities
    INTEGER(iwp), DIMENSION(:,:,:), ALLOCATABLE ::  topo_top_ind            !< precalculated topography top indices

    INTEGER(iwp), DIMENSION(:,:,:), ALLOCATABLE ::  topo_flags              !< flags to mask topography and surface-bounded grid
                                                                            !< points

    INTEGER(iwp), DIMENSION(:,:,:), POINTER ::  flags  !< pointer to topo_flags_1-10

    SAVE

 END MODULE indices


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Interfaces for special subroutines which use optional parameters.
!--------------------------------------------------------------------------------------------------!
 MODULE interfaces

    INTERFACE

!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> @todo Missing subroutine description.
!--------------------------------------------------------------------------------------------------!
       SUBROUTINE global_min_max ( i1, i2, j1, j2, k1, k2, array, mode, offset, result, result_ijk,&
                                   result1, result1_ijk )

          USE kinds

          CHARACTER (LEN=*), INTENT(IN) ::  mode                      !< mode of global min/max function: can be 'min', 'max',
                                                                      !< 'minmax', 'abs', or 'absoff'

          INTEGER(iwp), INTENT(IN)      ::  i1                        !< internal index of min/max function
          INTEGER(iwp), INTENT(IN)      ::  i2                        !< internal index of min/max function
          INTEGER(iwp), INTENT(IN)      ::  j1                        !< internal index of min/max function
          INTEGER(iwp), INTENT(IN)      ::  j2                        !< internal index of min/max function
          INTEGER(iwp), INTENT(IN)      ::  k1                        !< internal index of min/max function
          INTEGER(iwp), INTENT(IN)      ::  k2                        !< internal index of min/max function

          INTEGER(iwp)                  ::  result_ijk(3)             !< grid index result of min/max function
          INTEGER(iwp), OPTIONAL        ::  result1_ijk(3)            !< optional grid index result of min/max function

          REAL(wp)                      ::  offset                    !< min/max function calculates absolute value with respect to
                                                                      !< an offset
          REAL(wp)                      ::  result                    !< result of min/max function
          REAL(wp), OPTIONAL            ::  result1                   !< optional result of min/max function

          REAL(wp), INTENT(IN)          ::  array(i1:i2,j1:j2,k1:k2)  !< input array of min/max function

       END SUBROUTINE global_min_max

    END INTERFACE

    SAVE

 END MODULE interfaces


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Interfaces for subroutines with pointer arguments called in prognostic_equations.
!--------------------------------------------------------------------------------------------------!
 MODULE pointer_interfaces

    INTERFACE

!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> @todo Missing subroutine description.
!--------------------------------------------------------------------------------------------------!
       SUBROUTINE advec_s_bc( sk, sk_char )

          USE kinds

          CHARACTER (LEN=*), INTENT(IN) ::  sk_char  !< string for treated scalar in Bott-Chlond scheme

          REAL(wp), DIMENSION(:,:,:), POINTER ::  sk  !< treated scalar array in Bott-Chlond scheme

       END SUBROUTINE advec_s_bc

    END INTERFACE

    SAVE

 END MODULE pointer_interfaces


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Definition of variables which define processor topology and the exchange of ghost point layers.
!> This module must be placed in all routines containing MPI-calls.
!--------------------------------------------------------------------------------------------------!
 MODULE pegrid

    USE kinds

#if defined( __parallel )
    USE MPI,                                                                                       &
        ONLY: MPI_STATUS_SIZE

    PRIVATE MPI_STATUS_SIZE
#endif

    CHARACTER(LEN=2) ::  send_receive = 'al'     !<
    CHARACTER(LEN=7) ::  myid_char = ''          !< character string containing processor id number

    INTEGER(iwp) ::  comm1dx                     !< communicator for domain decomposition along x
    INTEGER(iwp) ::  comm1dy                     !< communicator for domain decomposition along y
    INTEGER(iwp) ::  comm2d                      !< standard 2d (xy) communicator used in PALM for the process group the PE belongs
                                                 !< to
    INTEGER(iwp) ::  comm_palm                   !< internal communicator used during the MPI setup at the beginning of a run
    INTEGER(iwp) ::  id_outflow = 0              !< myidx of procs at outflow (turbulent outflow method)
    INTEGER(iwp) ::  id_outflow_source = 0       !< myidx of procs including ouflow source plane (turbulent outflow method)
    INTEGER(iwp) ::  ierr                        !< standard error parameter in MPI calls
    INTEGER(iwp) ::  myid = 0                    !< id number of processor element
    INTEGER(iwp) ::  myidx = 0                   !< id number of processor elements with same position along x-direction
    INTEGER(iwp) ::  myidy = 0                   !< id number of processor elements with same position along y-direction
    INTEGER(iwp) ::  ndim = 2                    !< dimension of the virtual PE grid
    INTEGER(iwp) ::  ngp_xy                      !< used in atmosphere/ocean coupling: number of grid points of the subdomain
    INTEGER(iwp) ::  ngp_y                       !< number of subdomain grid points along y including ghost points
    INTEGER(iwp) ::  npex = -1                   !< number of processor elements in x-direction
    INTEGER(iwp) ::  npey = -1                   !< number of processor elements in y-direction
    INTEGER(iwp) ::  numprocs = 1                !< total number of appointed processor elements
    INTEGER(iwp) ::  numprocs_previous_run = -1  !< total number of appointed processor elements in previous run (job chain)
    INTEGER(iwp) ::  pleft                       !< MPI id of left neigbour pe
    INTEGER(iwp) ::  pnorth                      !< MPI id of right neigbour pe
    INTEGER(iwp) ::  pright                      !< MPI id of south neigbour pe
    INTEGER(iwp) ::  psouth                      !< MPI id of north neigbour pe
    INTEGER(iwp) ::  req_count = 0               !< MPI return variable - checks if Send-Receive operation is already finished
    INTEGER(iwp) ::  sendrecvcount_xy            !< number of subdomain gridpoints to be exchanged in direct transpositions
                                                 !< (y --> x, or x --> y) or second (2d) transposition x --> y
    INTEGER(iwp) ::  sendrecvcount_yz            !< number of subdomain gridpoints to be exchanged in third (2d) transposition
                                                 !< y --> z
    INTEGER(iwp) ::  sendrecvcount_zx            !< number of subdomain gridpoints to be exchanged in first (2d) transposition
                                                 !< z --> x
    INTEGER(iwp) ::  sendrecvcount_zyd           !< number of subdomain gridpoints to be exchanged in direct transpositions z --> y (used for calculating spectra)
    INTEGER(iwp) ::  target_id                   !< in atmosphere/ocean coupling: id of the ocean/atmosphere counterpart PE with
                                                 !< whom the atmosphere/ocean PE exchanges data
    INTEGER(iwp) ::  tasks_per_node = -9999      !< MPI tasks per compute node
    INTEGER(iwp) ::  threads_per_task = 1        !< number of OPENMP threads per MPI task
    INTEGER(iwp) ::  type_x                      !< derived MPI datatype for 2-D ghost-point exchange - north / south
    INTEGER(iwp) ::  type_xy                     !< derived MPI datatype for 2-D ghost-point exchange - north / south
    INTEGER(iwp) ::  type_y                      !< derived MPI datatype for 2-D exchange in atmosphere-ocean coupler

    INTEGER(iwp) ::  req(100)      !< MPI return variable indicating if send-receive operation is finished

    INTEGER(iwp), DIMENSION(:,:), ALLOCATABLE ::  hor_index_bounds               !< horizontal index bounds
    INTEGER(iwp), DIMENSION(:,:), ALLOCATABLE ::  hor_index_bounds_previous_run  !< horizontal index bounds of previous run

    LOGICAL ::  collective_wait = .FALSE.                 !< switch to set an explicit MPI barrier in front of all collective MPI calls
    LOGICAL ::  non_uniform_subdomain = .FALSE.           !< subdomains are non-uniform along x and/or y
    LOGICAL ::  non_uniform_data_for_transpose = .FALSE.  !< data to be transposed is non-uniformly distributed among the cores

    TYPE virtual_pe_grid
       INTEGER(iwp) ::  mpi_communicator         !< MPI communicator id
       INTEGER(iwp) ::  pleft                    !< MPI id of left neigbour pe
       INTEGER(iwp) ::  pright                   !< MPI id of right neigbour pe
       INTEGER(iwp) ::  psouth                   !< MPI id of south neigbour pe
       INTEGER(iwp) ::  pnorth                   !< MPI id of north neigbour pe
    END TYPE virtual_pe_grid

    TYPE(virtual_pe_grid) ::  communicator_configurations(4)  !< stores the four possible 2d virtual grids:
                                                              !< cyclic, cyclic along x, cyclic along y, non-cyclic

#if defined( __parallel )
    INTEGER(iwp) ::  ibuf(12)                 !< internal buffer for calculating MPI settings
    INTEGER(iwp) ::  pcoord(2)                !< PE coordinates along x and y
    INTEGER(iwp) ::  status(MPI_STATUS_SIZE)  !< MPI status variable used in various MPI calls
    INTEGER(iwp) ::  type_x_byte              !< derived MPI datatype for 2-D 8-bit integer ghost-point exchange - north / south
    INTEGER(iwp) ::  type_y_byte              !< derived MPI datatype for 2-D integer ghost-point exchange - left / right

    INTEGER(iwp), DIMENSION(MPI_STATUS_SIZE,100) ::  wait_stat  !< MPI status variable used in various MPI calls

    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  ngp_xz      !< number of ghost points in xz-plane on different multigrid level
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  ngp_xz_int  !< number of ghost points in xz-plane on different multigrid level
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  ngp_yz      !< number of ghost points in yz-plane on different multigrid level
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  ngp_yz_int  !< number of ghost points in yz-plane on different multigrid level
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  type_x_int  !< derived MPI datatype for 2-D integer ghost-point exchange - north /
                                                            !< south
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  type_xz     !< derived MPI datatype for 3-D integer ghost-point exchange - north /
                                                            !< south
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  type_xz_int !< derived MPI datatype for 3-D integer ghost-point exchange - north /
                                                            !< south
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  type_y_int  !< derived MPI datatype for 2-D integer ghost-point exchange - left /
                                                            !< right
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  type_yz     !< derived MPI datatype for 3-D integer ghost-point exchange - left /
                                                            !< right
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  type_yz_int !< derived MPI datatype for 3-D integer ghost-point exchange - left /
                                                            !< right

    LOGICAL ::  left_border_pe  = .FALSE.  !< = .TRUE. if PE is on left border of computational domain
    LOGICAL ::  north_border_pe = .FALSE.  !< = .TRUE. if PE is on north border of computational domain
    LOGICAL ::  reorder = .FALSE.          !< flag to allow MPI the reorder of ranking (e.g. row-major or column-major)
    LOGICAL ::  right_border_pe = .FALSE.  !< = .TRUE. if PE is on right border of computational domain
    LOGICAL ::  south_border_pe = .FALSE.  !< = .TRUE. if PE is on south border of computational domain

    LOGICAL, DIMENSION(2) ::  cyclic = (/ .TRUE. , .TRUE. /)  !< boundary conditions of the virtual PE grid
    LOGICAL, DIMENSION(2) ::  remain_dims                     !< internal array used to determine sub-topologies for transpositions
#endif

    SAVE

 END MODULE pegrid


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Definition of variables which control PROFIL-output.
!--------------------------------------------------------------------------------------------------!
 MODULE profil_parameter

    USE kinds

    INTEGER(iwp), PARAMETER ::  crmax = 100  !< maximum number of coordinate systems for profile output

    CHARACTER (LEN=27), DIMENSION(20) ::  cross_ts_profiles = &  !< time series to be plotted into one coordinate system, respectively
                                          (/ ' E E*                      ',                        &
                                             ' dt                        ',                        &
                                             ' u* w*                     ',                        &
                                             ' th*                       ',                        &
                                             ' umax vmax wmax            ',                        &
                                             ' div_old div_new           ',                        &
                                             ' zi_wtheta zi_theta        ',                        &
                                             ' w"theta"0 w"theta" wtheta ',                        &
                                             ' theta(0) theta(zp)        ',                        &
                                             ' splux spluy spluz         ',                        &
                                             ' L                         ',                        &
                                           ( '                           ', i9 = 1, 9 ) /)

    CHARACTER (LEN=100), DIMENSION(crmax) ::  cross_profiles = &  !< quantities to be plotted into one coordinate system, respectively
                                              (/ ' u v                                          ', &
                                                 ' pt                                           ', &
                                                 ' w"theta" w*theta* w*theta*BC wtheta wthetaBC ', &
                                                 ' w"u" w*u* wu w"v" w*v* wv                    ', &
                                                 ' km kh                                        ', &
                                                 ' l                                            ', &
                                 ( '                                              ', i9 = 1, 94 ) /)

    INTEGER(iwp) ::  profile_columns = 2  !< number of coordinate systems on a profile plot per column
    INTEGER(iwp) ::  profile_rows = 3     !< number of coordinate systems on a profile plot per row

    INTEGER(iwp) ::  dopr_index(500) = 0                !< index number of respective profile quantity
    INTEGER(iwp) ::  dopr_initial_index(500) = 0        !< index number of initial profiles to be output

    SAVE

 END MODULE profil_parameter

!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Definition of statistical quantities, e.g. global sums.
!--------------------------------------------------------------------------------------------------!
 MODULE statistics

    USE kinds

    CHARACTER (LEN=40) ::  region(0:9) = 'total domain                            '  !< label for statistic region

    INTEGER(iwp) ::  pr_max = 1000          !< maximum number of allowed output profiles
    INTEGER(iwp) ::  pr_palm = 200          !< number of output profiles
    INTEGER(iwp) ::  statistic_regions = 0  !< identifier for statistic regions
    INTEGER(iwp) ::  u_max_ijk(3) = -1      !< index values (i,j,k) of location where u_max occurs
    INTEGER(iwp) ::  v_max_ijk(3) = -1      !< index values (i,j,k) of location where v_max occurs
    INTEGER(iwp) ::  w_max_ijk(3) = -1      !< index values (i,j,k) of location where w_max occurs

    LOGICAL ::  flow_statistics_called = .FALSE.  !< flag that tells other routines if flow statistics was executed
                                                  !< (after each timestep)

    REAL(wp) ::  u_max = 0.0_wp  !< maximum of absolute u-veloctiy in entire domain
    REAL(wp) ::  v_max = 0.0_wp  !< maximum of absolute v-veloctiy in entire domain
    REAL(wp) ::  w_max = 0.0_wp  !< maximum of absolute w-veloctiy in entire domain

    REAL(wp), DIMENSION(2) ::  z_i  !< inversion height

    REAL(wp), DIMENSION(:), ALLOCATABLE ::  mean_surface_level_height  !< mean surface level height for the different statistic
                                                                       !< regions
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  sums_divnew_l              !< subdomain sum (_l) of divergence after pressure
                                                                       !< solver call (new)
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  sums_divold_l              !< subdomain sum (_l) of divergence before pressure
                                                                       !< solver call (old)
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  weight_pres                !< substep weighting factor for pressure solver
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  weight_substep             !< weighting factor for substeps in timestepping

    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums             !< global sum array for the various output quantities
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_ls_l        !< subdomain sum of large scale forcing and nudging tendencies
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_salsa_ws_l  !< subdomain sum of vertical salsa flux w's'
                                                               !< (5th-order advection scheme only)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_us2_ws_l    !< subdomain sum of horizontal momentum flux u'u'
                                                               !< (5th-order advection scheme only)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_vs2_ws_l    !< subdomain sum of horizontal momentum flux v'v'
                                                               !< (5th-order advection scheme only)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_ws2_ws_l    !< subdomain sum of vertical momentum flux w'w'
                                                               !< (5th-order advection scheme only)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_wsncs_ws_l  !< subdomain sum of vertical clouddrop-number concentration flux
                                                               !< w'nc' (5th-order advection scheme only)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_wsngs_ws_l  !< subdomain sum of vertical graupel-number concentration flux w'nc' (5th-order advection scheme only)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_wsnis_ws_l  !< subdomain sum of vertical ice crystal concentration flux w'ni'
                                                               !< (5th-order advection scheme only)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_wsnrs_ws_l  !< subdomain sum of vertical raindrop-number concentration flux w'nr'
                                                               !< (5th-order advection scheme only)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_wsnss_ws_l  !< subdomain sum of vertical snow-number concentration flux w'ns' (5th-order advection scheme only)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_wssas_ws_l  !< subdomain sum of vertical salinity flux w'sa'
                                                               !< (5th-order advection scheme only)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_wsss_ws_l   !< subdomain sum of vertical passive scalar flux w's'
                                                               !< (5th-order advection scheme only)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_wspts_ws_l  !< subdomain sum of vertical sensible heat flux w'pt'
                                                               !< (5th-order advection scheme only)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_wsqcs_ws_l  !< subdomain sum of vertical cloudwater flux w'qc'
                                                               !< (5th-order advection scheme only)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_wsqgs_ws_l  !< subdomain sum of vertical graupel flux w'qg' (5th-order advection scheme only)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_wsqis_ws_l  !< subdomain sum of vertical ice crystal flux w'qi'
                                                               !< (5th-order advection scheme only)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_wsqrs_ws_l  !< subdomain sum of vertical rainwater flux w'qr'
                                                               !< (5th-order advection scheme only)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_wsqss_ws_l  !< subdomain sum of vertical snow flux w'qs' (5th-order advection scheme only)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_wsqs_ws_l   !< subdomain sum of vertical latent heat flux w'q'
                                                               !< (5th-order advection scheme only)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_wsts_bc_l   !< subdomain sum of sensible heat flux in Bott-Chlond scheme
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_wsus_ws_l   !< subdomain sum of vertical momentum flux w'u'
                                                               !< (5th-order advection scheme only)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  sums_wsvs_ws_l   !< subdomain sum of vertical momentum flux w'v'
                                                               !< (5th-order advection scheme only)
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  ts_value         !< timeseries output array for the various output quantities

    REAL(wp), DIMENSION(:,:), POINTER     ::  sums_wschs_ws_l  !< subdomain sum of vertical chemistry flux w'ch'

    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  hom_sum             !< sum array for horizontal mean
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  rmask               !< REAL flag array (0.0 or 1.0) for statistic regions
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  sums_l              !< subdomain sum (_l) gathered for various quantities
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::  sums_l_l            !< subdomain sum (_l) of mixing length from diffusivities

    REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE ::  hom  !< horizontal mean of various quantities (profiles/timeseries)

    SAVE

 END MODULE statistics
