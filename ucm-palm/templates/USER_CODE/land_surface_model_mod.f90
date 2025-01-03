!> @file land_surface_model_mod.f90
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
!> Land surface model, consisting of a solver for the energy balance at the surface and a multi
!> layer soil scheme. The scheme is similar to the TESSEL scheme implemented in the ECMWF IFS model,
!> with modifications according to H-TESSEL. The implementation is based on the formulation
!> implemented in the DALES and UCLA-LES models.
!>
!> @todo Extensive verification energy-balance solver for vertical surfaces,
!>       e.g. parametrization of r_a
!> @todo Revise single land-surface processes for vertical surfaces, e.g. treatment of humidity,
!>       etc.
!> @todo Consider partial absorption of the net shortwave radiation by the skin layer.
!> @todo Improve surface water parameterization
!> @todo Invert indices (running from -3 to 0. Currently: nzb_soil=0, nzt_soil=3)).
!> @todo Implement surface runoff model (required when performing long-term LES with considerable
!>       precipitation.
!> @todo Revise calculation of f2 when wilting point is non-constant in the soil
!> @todo Allow for zero soil moisture (currently, it is set to wilting point)
!> @note No time step criterion is required as long as the soil layers do not become too thin.
!> @todo Attention, pavement_subpars_1/2 are hardcoded to 8 levels, in case more levels are used
!>       this may cause an potential bug
!> @todo Routine calc_q_surface required?
!> @todo Allow for precipitation water to enter pavements that are semi-pervious
!--------------------------------------------------------------------------------------------------!
 MODULE land_surface_model_mod

#if defined( __parallel )
    USE MPI
#endif

    USE arrays_3d,                                                                                 &
        ONLY:  d_exner,                                                                            &
               exner,                                                                              &
               hyp,                                                                                &
               hyrho,                                                                              &
               pt,                                                                                 &
               prr,                                                                                &
               q,                                                                                  &
               ql,                                                                                 &
               vpt,                                                                                &
               u,                                                                                  &
               v,                                                                                  &
               w

    USE basic_constants_and_equations_mod,                                                         &
        ONLY:  c_p,                                                                                &
               g,                                                                                  &
               lv_d_cp,                                                                            &
               l_v,                                                                                &
               magnus,                                                                             &
               rho_l,                                                                              &
               r_d,                                                                                &
               rd_d_rv

    USE calc_mean_profile_mod,                                                                     &
        ONLY:  calc_mean_profile

    USE control_parameters,                                                                        &
        ONLY:  allow_roughness_limitation,                                                         &
               cloud_droplets,                                                                     &
               coupling_char,                                                                      &
               cyclic_fill_initialization,                                                         &
               dcep,                                                                               &
               debug_output,                                                                       &
               debug_output_timestep,                                                              &
               debug_string,                                                                       &
               dt_3d,                                                                              &
               external_anthropogenic_heat,                                                        &
               humidity,                                                                           &
               initializing_actions,                                                               &
               intermediate_timestep_count,                                                        &
               intermediate_timestep_count_max,                                                    &
               land_surface,                                                                       &
               read_spinup_data,                                                                   &
               restart_data_format_output,                                                         &
               rho_surface,                                                                        &
               surface_pressure,                                                                   &
               timestep_scheme,                                                                    &
               tsc

    USE cpulog,                                                                                    &
        ONLY:  cpu_log,                                                                            &
               log_point_s

    USE indices,                                                                                   &
        ONLY:  nbgp,                                                                               &
               nxl,                                                                                &
               nxlg,                                                                               &
               nxr,                                                                                &
               nxrg,                                                                               &
               nyn,                                                                                &
               nyng,                                                                               &
               nys,                                                                                &
               nysg,                                                                               &
               nzb

    USE bulk_cloud_model_mod,                                                                      &
        ONLY: bulk_cloud_model,                                                                    &
              precipitation

    USE netcdf_data_input_mod,                                                                     &
        ONLY :  albedo_type_f,                                                                     &
                building_type_f,                                                                   &
                char_fill,                                                                         &
                char_lod,                                                                          &
                check_existence,                                                                   &
                close_input_file,                                                                  &
                get_attribute,                                                                     &
                get_dimension_length,                                                              &
                get_variable,                                                                      &
                init_3d,                                                                           &
                input_file_dynamic,                                                                &
                input_pids_dynamic,                                                                &
                input_pids_static,                                                                 &
                inquire_num_variables,                                                             &
                inquire_variable_names,                                                            &
                num_var_pids,                                                                      &
                open_read_file,                                                                    &
                pids_id,                                                                           &
                pavement_pars_f,                                                                   &
                pavement_subsurface_pars_f,                                                        &
                pavement_type_f,                                                                   &
                root_area_density_lsm_f,                                                           &
                soil_pars_f,                                                                       &
                soil_type_f,                                                                       &
                surface_fraction_f,                                                                &
                vars_pids,                                                                         &
                vegetation_pars_f,                                                                 &
                vegetation_type_f,                                                                 &
                water_pars_f,                                                                      &
                water_type_f

    USE kinds

    USE pegrid

    USE radiation_model_mod,                                                                       &
        ONLY:  albedo,                                                                             &
               albedo_type,                                                                        &
               emissivity,                                                                         &
               force_radiation_call,                                                               &
               radiation,                                                                          &
               radiation_scheme,                                                                   &
               unscheduled_radiation_calls

    USE restart_data_mpi_io_mod,                                                                   &
        ONLY:  rd_mpi_io_check_array,                                                              &
               rd_mpi_io_surface_filetypes,                                                        &
               rrd_mpi_io,                                                                         &
               rrd_mpi_io_surface,                                                                 &
               wrd_mpi_io, wrd_mpi_io_surface

    USE statistics,                                                                                &
        ONLY:  hom,                                                                                &
               statistic_regions,                                                                  &
               ts_value

    USE surface_mod,                                                                               &
        ONLY:  ind_pav_green,                                                                      &
               ind_veg_wall,                                                                       &
               ind_wat_win,                                                                        &
               soil_moisture,                                                                      &
               surf_lsm,                                                                           &
               surf_type,                                                                          &
               surface_restore_elements

    IMPLICIT NONE

    TYPE surf_type_lsm
       REAL(wp), DIMENSION(:),   ALLOCATABLE ::  var_1d !< 1D prognostic variable
       REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  var_2d !< 2D prognostic variable
    END TYPE surf_type_lsm

!
!-- LSM model constants
    REAL(wp), PARAMETER ::  b_ch           = 6.04_wp   ! Clapp & Hornberger exponent
    REAL(wp), PARAMETER ::  lambda_h_dry   = 0.19_wp   ! heat conductivity for dry soil (W/m/K)
    REAL(wp), PARAMETER ::  lambda_h_sm    = 3.44_wp   ! heat conductivity of the soil matrix (W/m/K)
    REAL(wp), PARAMETER ::  lambda_h_water = 0.57_wp   ! heat conductivity of water (W/m/K)
    REAL(wp), PARAMETER ::  m_max_depth    = 0.0002_wp ! Maximum capacity of the water reservoir on a flat surface (leaf/bare soil) (m)
    REAL(wp), PARAMETER ::  psi_sat        = -0.388_wp ! soil matrix potential at saturation
    REAL(wp), PARAMETER ::  rho_c_soil     = 2.19E6_wp ! volumetric heat capacity of soil (J/m3/K)
    REAL(wp), PARAMETER ::  rho_c_water    = 4.20E6_wp  ! volumetric heat capacity of water (J/m3/K)


    REAL(wp), DIMENSION(0:7), PARAMETER  :: dz_soil_default =                                      & !< default soil layer configuration
                                            (/ 0.01_wp, 0.02_wp, 0.04_wp,                          &
                                               0.06_wp, 0.14_wp, 0.26_wp,                          &
                                               0.54_wp, 1.86_wp /)

    REAL(wp), DIMENSION(0:3), PARAMETER  :: dz_soil_ref = (/ 0.07_wp, 0.21_wp, 0.72_wp, 1.89_wp /) !< reference four layer soil configuration

    REAL(wp), DIMENSION(0:3), PARAMETER  :: zs_ref = (/ 0.07_wp, 0.28_wp, 1.0_wp, 2.89_wp /) !< used for estimating the root fractions



!
!-- LSM variables
    CHARACTER(14) :: surface_type = 'read_from_file' !< general classification. Allowed are:
                                                     !< 'vegetation', 'pavement', ('building'), 'water',
                                                     !< 'read_from_file' and 'netcdf' (deprecated).

    INTEGER(iwp) ::  npavement_subsurface_pars = 2   !< number of pavement_subsurface_pars
    INTEGER(iwp) ::  nzt_pavement = 0                !< top of the pavement within the soil
    INTEGER(iwp) ::  nzb_soil = 0                    !< bottom of the soil model (Earth's surface)
    INTEGER(iwp) ::  nzt_soil = 7                    !< top of the soil model
    INTEGER(iwp) ::  nzs = 8                         !< number of soil layers
    INTEGER(iwp) ::  pavement_depth_level = 0        !< default NAMELIST nzt_pavement
    INTEGER(iwp) ::  pavement_type = 1               !< default NAMELIST pavement_type
    INTEGER(iwp) ::  soil_type = 3                   !< default NAMELIST soil_type
    INTEGER(iwp) ::  vegetation_type = 2             !< default NAMELIST vegetation_type
    INTEGER(iwp) ::  water_type = 1                  !< default NAMELISt water_type


    LOGICAL ::  aero_resist_kray = .TRUE.         !< flag to control parametrization of aerodynamic resistance at vertical surface elements
    LOGICAL ::  conserve_water_content = .TRUE.   !< open or closed bottom surface for the soil model
    LOGICAL ::  constant_roughness = .FALSE.      !< use fixed/dynamic roughness lengths for water surfaces
    LOGICAL ::  force_radiation_call_l = .FALSE.  !< flag to force calling of radiation routine

!   value 9999999.9_wp -> generic available or user-defined value must be set
!   otherwise -> no generic variable and user setting is optional
    REAL(wp) ::  alpha_vangenuchten = 9999999.9_wp             !< NAMELIST alpha_vg
    REAL(wp) ::  canopy_resistance_coefficient = 9999999.9_wp  !< NAMELIST g_d
    REAL(wp) ::  c_surface = 9999999.9_wp                      !< Surface (skin) heat capacity (J/m2/K)
    REAL(wp) ::  deep_soil_temperature =  9999999.9_wp         !< Deep soil temperature (bottom boundary condition)
    REAL(wp) ::  drho_l_lv                                     !< (rho_l * l_v)**-1
    REAL(wp) ::  dt_lsm = HUGE( 1.0_wp )                       !< maximum allowed timestep of the land-surface model
    REAL(wp) ::  field_capacity = 9999999.9_wp                 !< NAMELIST m_fc
    REAL(wp) ::  f_shortwave_incoming = 9999999.9_wp           !< NAMELIST f_sw_in
    REAL(wp) ::  hydraulic_conductivity = 9999999.9_wp         !< NAMELIST gamma_w_sat
    REAL(wp) ::  ke = 0.0_wp                                   !< Kersten number
    REAL(wp) ::  lambda_h_sat = 0.0_wp                         !< heat conductivity for saturated soil (W/m/K)
    REAL(wp) ::  lambda_surface_stable = 9999999.9_wp          !< NAMELIST lambda_surface_s (W/m2/K)
    REAL(wp) ::  lambda_surface_unstable = 9999999.9_wp        !< NAMELIST lambda_surface_u (W/m2/K)
    REAL(wp) ::  leaf_area_index = 9999999.9_wp                !< NAMELIST lai
    REAL(wp) ::  l_vangenuchten = 9999999.9_wp                 !< NAMELIST l_vg
    REAL(wp) ::  min_canopy_resistance = 9999999.9_wp          !< NAMELIST r_canopy_min
    REAL(wp) ::  min_soil_resistance = 50.0_wp                 !< NAMELIST r_soil_min
    REAL(wp) ::  m_total = 0.0_wp                              !< weighted total water content of the soil (m3/m3)
    REAL(wp) ::  n_vangenuchten = 9999999.9_wp                 !< NAMELIST n_vg
    REAL(wp) ::  pavement_heat_capacity = 9999999.9_wp         !< volumetric heat capacity of pavement (e.g. roads) (J/m3/K)
    REAL(wp) ::  pavement_heat_conduct  = 9999999.9_wp         !< heat conductivity for pavements (e.g. roads) (W/m/K)
    REAL(wp) ::  q_s = 0.0_wp                                  !< saturation water vapor mixing ratio
    REAL(wp) ::  residual_moisture = 9999999.9_wp              !< NAMELIST m_res
    REAL(wp) ::  rho_cp                                        !< rho_surface * cp
    REAL(wp) ::  rho_lv                                        !< rho_ocean * l_v
    REAL(wp) ::  saturation_moisture = 9999999.9_wp            !< NAMELIST m_sat
    REAL(wp) ::  skip_time_do_lsm = 0.0_wp                     !< LSM is not called before this time
    REAL(wp) ::  vegetation_coverage = 9999999.9_wp            !< NAMELIST c_veg
    REAL(wp) ::  water_temperature = 9999999.9_wp              !< water temperature
    REAL(wp) ::  wilting_point = 9999999.9_wp                  !< NAMELIST m_wilt
    REAL(wp) ::  z0_vegetation  = 9999999.9_wp                 !< NAMELIST z0 (lsm_par)
    REAL(wp) ::  z0h_vegetation = 9999999.9_wp                 !< NAMELIST z0h (lsm_par)
    REAL(wp) ::  z0q_vegetation = 9999999.9_wp                 !< NAMELIST z0q (lsm_par)
    REAL(wp) ::  z0_pavement    = 9999999.9_wp                 !< NAMELIST z0 (lsm_par)
    REAL(wp) ::  z0h_pavement   = 9999999.9_wp                 !< NAMELIST z0h (lsm_par)
    REAL(wp) ::  z0q_pavement   = 9999999.9_wp                 !< NAMELIST z0q (lsm_par)
    REAL(wp) ::  z0_water       = 9999999.9_wp                 !< NAMELIST z0 (lsm_par)
    REAL(wp) ::  z0h_water      = 9999999.9_wp                 !< NAMELIST z0h (lsm_par)
    REAL(wp) ::  z0q_water      = 9999999.9_wp                 !< NAMELIST z0q (lsm_par)

    REAL(wp), DIMENSION(:), ALLOCATABLE ::  ddz_soil           !< 1/dz_soil
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  ddz_soil_center    !< 1/dz_soil_center
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  dz_soil_center     !< soil grid spacing (center-center)
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  root_extr          !< root extraction
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  zs                 !< depth of the temperature/moisute levels

    REAL(wp), DIMENSION(0:20) ::  dz_soil  = 9999999.9_wp         !< NAMELIST soil layer depths (spacing)
    REAL(wp), DIMENSION(0:20) ::  root_fraction = 9999999.9_wp    !< NAMELIST distribution of root surface area to the individual soil layers
    REAL(wp), DIMENSION(0:20) ::  soil_temperature = 9999999.9_wp !< NAMELIST soil temperature (K) +1
    REAL(wp), DIMENSION(0:20) ::  zs_layer = 9999999.9_wp         !< soil layer depths (edge)

    TYPE(surf_type_lsm), POINTER ::  m_soil   !< Soil moisture (m3/m3)
    TYPE(surf_type_lsm), POINTER ::  m_soil_p !< Prog. soil moisture (m3/m3)
    TYPE(surf_type_lsm), POINTER ::  t_soil   !< Soil temperature (K)
    TYPE(surf_type_lsm), POINTER ::  t_soil_p !< Prog. soil temperature (K)

    TYPE(surf_type_lsm), TARGET ::  m_soil_1   !< target array for soil moisture used to change timelevels
    TYPE(surf_type_lsm), TARGET ::  m_soil_2   !< target array for soil moisture used to change timelevels
    TYPE(surf_type_lsm), TARGET ::  t_soil_1   !< target array for soil temperature used to change timelevels
    TYPE(surf_type_lsm), TARGET ::  t_soil_2   !< target array for soil temperature used to change timelevels

    TYPE(surf_type_lsm), POINTER ::  m_liq        !< liquid water reservoir (m)
    TYPE(surf_type_lsm), POINTER ::  m_liq_p      !< progn. liquid water reservoir (m)
    TYPE(surf_type_lsm), POINTER ::  t_surface    !< surface temperature (K)
    TYPE(surf_type_lsm), POINTER ::  t_surface_p  !< progn. surface temperature (K)

    TYPE(surf_type_lsm), TARGET ::  m_liq_1       !< target array for liquid water reservoir used to change timelevels
    TYPE(surf_type_lsm), TARGET ::  m_liq_2       !< target array for liquid water reservoir used to change timelevels
    TYPE(surf_type_lsm), TARGET ::  t_surface_1   !< target array for surface temperature reservoir used to change timelevels
    TYPE(surf_type_lsm), TARGET ::  t_surface_2   !< target array for surface temperature reservoir used to change timelevels

    REAL(wp), DIMENSION(:,:), ALLOCATABLE, TARGET :: m_liq_av !< liquid water reservoir - temporally averaged

    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  m_soil_av !< soil moisture - temporally averaged
    REAL(wp), DIMENSION(:,:,:), ALLOCATABLE, TARGET ::  t_soil_av !< soil temperature - temporally averaged

    TYPE(surf_type_lsm), TARGET ::  tm_liq_m      !< liquid water reservoir tendency (m)
    TYPE(surf_type_lsm), TARGET ::  tm_soil_m     !< m_soil storage array
    TYPE(surf_type_lsm), TARGET ::  tt_soil_m     !< t_soil storage array
    TYPE(surf_type_lsm), TARGET ::  tt_surface_m  !< surface temperature tendency (K)

!
!-- Energy balance variables - temporally averaged
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  c_liq_av      !< average of c_liq
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  c_soil_av     !< average of c_soil
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  c_veg_av      !< average of c_veg
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  lai_av        !< average of lai
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  qsws_liq_av   !< average of qsws_liq
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  qsws_soil_av  !< average of qsws_soil
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  qsws_veg_av   !< average of qsws_veg
    REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  r_s_av        !< average of r_s


!
!-- Predefined Land surface classes (vegetation_type)
    CHARACTER(26), DIMENSION(0:18), PARAMETER ::  vegetation_type_name = (/                        &
                                                     'user defined              ',                 & !  0
                                                     'bare soil                 ',                 & !  1
                                                     'crops, mixed farming      ',                 & !  2
                                                     'short grass               ',                 & !  3
                                                     'evergreen needleleaf trees',                 & !  4
                                                     'deciduous needleleaf trees',                 & !  5
                                                     'evergreen broadleaf trees ',                 & !  6
                                                     'deciduous broadleaf trees ',                 & !  7
                                                     'tall grass                ',                 & !  8
                                                     'desert                    ',                 & !  9
                                                     'tundra                    ',                 & ! 10
                                                     'irrigated crops           ',                 & ! 11
                                                     'semidesert                ',                 & ! 12
                                                     'ice caps and glaciers     ',                 & ! 13
                                                     'bogs and marshes          ',                 & ! 14
                                                     'evergreen shrubs          ',                 & ! 15
                                                     'deciduous shrubs          ',                 & ! 16
                                                     'mixed forest/woodland     ',                 & ! 17
                                                     'interrupted forest        '                  & ! 18
                                                                         /)

!
!-- Soil model classes (soil_type)
    CHARACTER(12), DIMENSION(0:6), PARAMETER ::  soil_type_name = (/                               &
                                                     'user defined',                               & ! 0
                                                     'coarse      ',                               & ! 1
                                                     'medium      ',                               & ! 2
                                                     'medium-fine ',                               & ! 3
                                                     'fine        ',                               & ! 4
                                                     'very fine   ',                               & ! 5
                                                     'organic     '                                & ! 6
                                                                  /)

!
!-- Pavement classes
    CHARACTER(29), DIMENSION(0:15), PARAMETER ::  pavement_type_name = (/                          &
                                                     'user defined                 ',              & ! 0
                                                     'asphalt/concrete mix         ',              & ! 1
                                                     'asphalt (asphalt concrete)   ',              & ! 2
                                                     'concrete (Portland concrete) ',              & ! 3
                                                     'sett                         ',              & ! 4
                                                     'paving stones                ',              & ! 5
                                                     'cobblestone                  ',              & ! 6
                                                     'metal                        ',              & ! 7
                                                     'wood                         ',              & ! 8
                                                     'gravel                       ',              & ! 9
                                                     'fine gravel                  ',              & ! 10
                                                     'pebblestone                  ',              & ! 11
                                                     'woodchips                    ',              & ! 12
                                                     'tartan (sports)              ',              & ! 13
                                                     'artifical turf (sports)      ',              & ! 14
                                                     'clay (sports)                '               & ! 15
                                                                        /)

!
!-- Water classes
    CHARACTER(12), DIMENSION(0:5), PARAMETER ::  water_type_name = (/                              &
                                                     'user defined',                               & ! 0
                                                     'lake        ',                               & ! 1
                                                     'river       ',                               & ! 2
                                                     'ocean       ',                               & ! 3
                                                     'pond        ',                               & ! 4
                                                     'fountain    '                                & ! 5
                                                                    /)


!
!-- Output parameter
    INTEGER(iwp) ::  dots_start_index_lsm  !< start index for time series of this module

!
!-- Land surface parameters according to the respective classes (vegetation_type)
    INTEGER(iwp) ::  ind_v_rc_min = 0    !< index for r_canopy_min in vegetation_pars
    INTEGER(iwp) ::  ind_v_rc_lai = 1    !< index for LAI in vegetation_pars
    INTEGER(iwp) ::  ind_v_c_veg   = 2   !< index for c_veg in vegetation_pars
    INTEGER(iwp) ::  ind_v_gd  = 3       !< index for g_d in vegetation_pars
    INTEGER(iwp) ::  ind_v_z0 = 4        !< index for z0 in vegetation_pars
    INTEGER(iwp) ::  ind_v_z0qh = 5      !< index for z0h / z0q in vegetation_pars
    INTEGER(iwp) ::  ind_v_lambda_s = 6  !< index for lambda_s_s in vegetation_pars
    INTEGER(iwp) ::  ind_v_lambda_u = 7  !< index for lambda_s_u in vegetation_pars
    INTEGER(iwp) ::  ind_v_f_sw_in = 8   !< index for f_sw_in in vegetation_pars
    INTEGER(iwp) ::  ind_v_c_surf = 9    !< index for c_surface in vegetation_pars
    INTEGER(iwp) ::  ind_v_at = 10       !< index for albedo_type in vegetation_pars
    INTEGER(iwp) ::  ind_v_emis = 11     !< index for emissivity in vegetation_pars

    INTEGER(iwp) ::  ind_w_temp     = 0    !< index for temperature in water_pars
    INTEGER(iwp) ::  ind_w_z0       = 1    !< index for z0 in water_pars
    INTEGER(iwp) ::  ind_w_z0h      = 2    !< index for z0h in water_pars
    INTEGER(iwp) ::  ind_w_lambda_s = 3    !< index for lambda_s_s in water_pars
    INTEGER(iwp) ::  ind_w_lambda_u = 4    !< index for lambda_s_u in water_pars
    INTEGER(iwp) ::  ind_w_at       = 5    !< index for albedo type in water_pars
    INTEGER(iwp) ::  ind_w_emis     = 6    !< index for emissivity in water_pars

    INTEGER(iwp) ::  ind_p_z0       = 0    !< index for z0 in pavement_pars
    INTEGER(iwp) ::  ind_p_z0h      = 1    !< index for z0h in pavement_pars
    INTEGER(iwp) ::  ind_p_at       = 2    !< index for albedo type in pavement_pars
    INTEGER(iwp) ::  ind_p_emis     = 3    !< index for emissivity in pavement_pars
    INTEGER(iwp) ::  ind_p_lambda_h = 0    !< index for lambda_h in pavement_subsurface_pars
    INTEGER(iwp) ::  ind_p_rho_c    = 1    !< index for rho_c in pavement_pars
!
!-- Land surface parameters
!-- r_canopy_min,     lai,   c_veg,     g_d         z0,         z0h, lambda_s_s, lambda_s_u, f_sw_in,  c_surface, albedo_type, emissivity
    REAL(wp), DIMENSION(0:11,1:18), PARAMETER :: vegetation_pars = RESHAPE( (/                                                  &
          0.0_wp, 0.00_wp, 0.00_wp, 0.00_wp,  0.005_wp,   0.5E-4_wp,     0.0_wp,    0.0_wp, 0.00_wp, 0.00_wp, 17.0_wp, 0.94_wp, & !  1
        180.0_wp, 3.00_wp, 1.00_wp, 0.00_wp,   0.10_wp,    0.001_wp,    10.0_wp,   10.0_wp, 0.05_wp, 0.00_wp,  2.0_wp, 0.95_wp, & !  2
        110.0_wp, 2.00_wp, 1.00_wp, 0.00_wp,   0.03_wp,   0.3E-4_wp,    10.0_wp,   10.0_wp, 0.05_wp, 0.00_wp,  5.0_wp, 0.95_wp, & !  3
        500.0_wp, 5.00_wp, 1.00_wp, 0.03_wp,   2.00_wp,     2.00_wp,    20.0_wp,   15.0_wp, 0.03_wp, 0.00_wp,  6.0_wp, 0.97_wp, & !  4
        500.0_wp, 5.00_wp, 1.00_wp, 0.03_wp,   2.00_wp,     2.00_wp,    20.0_wp,   15.0_wp, 0.03_wp, 0.00_wp,  8.0_wp, 0.97_wp, & !  5
        175.0_wp, 5.00_wp, 1.00_wp, 0.03_wp,   2.00_wp,     2.00_wp,    20.0_wp,   15.0_wp, 0.03_wp, 0.00_wp,  9.0_wp, 0.97_wp, & !  6
        240.0_wp, 6.00_wp, 0.99_wp, 0.03_wp,   2.00_wp,     2.00_wp,    20.0_wp,   15.0_wp, 0.03_wp, 0.00_wp,  7.0_wp, 0.97_wp, & !  7
        100.0_wp, 2.00_wp, 0.70_wp, 0.00_wp,   0.47_wp,  0.47E-2_wp,    10.0_wp,   10.0_wp, 0.05_wp, 0.00_wp, 10.0_wp, 0.97_wp, & !  8
        250.0_wp, 0.05_wp, 0.00_wp, 0.00_wp,  0.013_wp, 0.013E-2_wp,    15.0_wp,   15.0_wp, 0.00_wp, 0.00_wp, 11.0_wp, 0.94_wp, & !  9
         80.0_wp, 1.00_wp, 0.50_wp, 0.00_wp,  0.034_wp, 0.034E-2_wp,    10.0_wp,   10.0_wp, 0.05_wp, 0.00_wp, 13.0_wp, 0.97_wp, & ! 10
        180.0_wp, 3.00_wp, 1.00_wp, 0.00_wp,    0.5_wp,  0.50E-2_wp,    10.0_wp,   10.0_wp, 0.05_wp, 0.00_wp,  2.0_wp, 0.97_wp, & ! 11
        150.0_wp, 0.50_wp, 0.10_wp, 0.00_wp,   0.17_wp,  0.17E-2_wp,    10.0_wp,   10.0_wp, 0.05_wp, 0.00_wp, 11.0_wp, 0.97_wp, & ! 12
          0.0_wp, 0.00_wp, 0.00_wp, 0.00_wp, 1.3E-3_wp,   1.3E-4_wp,    58.0_wp,   58.0_wp, 0.00_wp, 0.00_wp, 14.0_wp, 0.97_wp, & ! 13
        240.0_wp, 4.00_wp, 0.60_wp, 0.00_wp,   0.83_wp,  0.83E-2_wp,    10.0_wp,   10.0_wp, 0.05_wp, 0.00_wp,  3.0_wp, 0.97_wp, & ! 14
        225.0_wp, 3.00_wp, 0.50_wp, 0.00_wp,   0.10_wp,  0.10E-2_wp,    10.0_wp,   10.0_wp, 0.05_wp, 0.00_wp,  4.0_wp, 0.97_wp, & ! 15
        225.0_wp, 1.50_wp, 0.50_wp, 0.00_wp,   0.25_wp,  0.25E-2_wp,    10.0_wp,   10.0_wp, 0.05_wp, 0.00_wp,  5.0_wp, 0.97_wp, & ! 16
        250.0_wp, 5.00_wp, 1.00_wp, 0.03_wp,   2.00_wp,     2.00_wp,    20.0_wp,   15.0_wp, 0.03_wp, 0.00_wp, 10.0_wp, 0.97_wp, & ! 17
        175.0_wp, 2.50_wp, 1.00_wp, 0.03_wp,   1.10_wp,     1.10_wp,    20.0_wp,   15.0_wp, 0.03_wp, 0.00_wp,  7.0_wp, 0.97_wp  & ! 18
                                                                             /), (/ 12, 18 /) )


!
!-- Root distribution for default soil layer configuration (sum = 1)
!--                                level 1 - level 4 according to zs_ref
    REAL(wp), DIMENSION(0:3,1:18), PARAMETER :: root_distribution = RESHAPE( (/                    &
                                                1.00_wp, 0.00_wp, 0.00_wp, 0.00_wp,                & !  1
                                                0.24_wp, 0.41_wp, 0.31_wp, 0.04_wp,                & !  2
                                                0.35_wp, 0.38_wp, 0.23_wp, 0.04_wp,                & !  3
                                                0.26_wp, 0.39_wp, 0.29_wp, 0.06_wp,                & !  4
                                                0.26_wp, 0.38_wp, 0.29_wp, 0.07_wp,                & !  5
                                                0.24_wp, 0.38_wp, 0.31_wp, 0.07_wp,                & !  6
                                                0.25_wp, 0.34_wp, 0.27_wp, 0.14_wp,                & !  7
                                                0.27_wp, 0.27_wp, 0.27_wp, 0.09_wp,                & !  8
                                                1.00_wp, 0.00_wp, 0.00_wp, 0.00_wp,                & !  9
                                                0.47_wp, 0.45_wp, 0.08_wp, 0.00_wp,                & ! 10
                                                0.24_wp, 0.41_wp, 0.31_wp, 0.04_wp,                & ! 11
                                                0.17_wp, 0.31_wp, 0.33_wp, 0.19_wp,                & ! 12
                                                0.00_wp, 0.00_wp, 0.00_wp, 0.00_wp,                & ! 13
                                                0.25_wp, 0.34_wp, 0.27_wp, 0.11_wp,                & ! 14
                                                0.23_wp, 0.36_wp, 0.30_wp, 0.11_wp,                & ! 15
                                                0.23_wp, 0.36_wp, 0.30_wp, 0.11_wp,                & ! 16
                                                0.19_wp, 0.35_wp, 0.36_wp, 0.10_wp,                & ! 17
                                                0.19_wp, 0.35_wp, 0.36_wp, 0.10_wp                 & ! 18
                                                                               /), (/ 4, 18 /) )

!
!-- Soil parameters according to the following porosity classes (soil_type)

!
!-- Soil parameters  alpha_vg,      l_vg,    n_vg, gamma_w_sat,    m_sat,     m_fc,   m_wilt,    m_res
    REAL(wp), DIMENSION(0:7,1:6), PARAMETER :: soil_pars = RESHAPE( (/                        &
             3.83_wp,  1.250_wp, 1.38_wp,  6.94E-6_wp, 0.403_wp, 0.244_wp, 0.059_wp, 0.025_wp,& ! 1
             3.14_wp, -2.342_wp, 1.28_wp,  1.16E-6_wp, 0.439_wp, 0.347_wp, 0.151_wp, 0.010_wp,& ! 2
             0.83_wp, -0.588_wp, 1.25_wp,  0.26E-6_wp, 0.430_wp, 0.383_wp, 0.133_wp, 0.010_wp,& ! 3
             3.67_wp, -1.977_wp, 1.10_wp,  2.87E-6_wp, 0.520_wp, 0.448_wp, 0.279_wp, 0.010_wp,& ! 4
             2.65_wp,  2.500_wp, 1.10_wp,  1.74E-6_wp, 0.614_wp, 0.541_wp, 0.335_wp, 0.010_wp,& ! 5
             1.30_wp,  0.400_wp, 1.20_wp,  0.93E-6_wp, 0.766_wp, 0.663_wp, 0.267_wp, 0.010_wp & ! 6
                                                                     /), (/ 8, 6 /) )


!
!-- TO BE FILLED
!-- Pavement parameters      z0,       z0h, albedo_type, emissivity
    REAL(wp), DIMENSION(0:3,1:15), PARAMETER :: pavement_pars = RESHAPE( (/                   &
                                              5.0E-2_wp, 5.0E-4_wp,     18.0_wp,    0.93_wp,  & !  1 z0: Masson et al. (2002), eps: Mix of asphalt and concrete
                                              5.0E-2_wp, 5.0E-4_wp,     19.0_wp,    0.95_wp,  & !  2 z0: Masson et al. (2002), eps: Oke (1987)
                                              5.0E-2_wp, 5.0E-4_wp,     20.0_wp,    0.90_wp,  & !  3 z0: Masson et al. (2002), eps: Yaghoobian et al. (2009)
                                              5.0E-2_wp, 5.0E-4_wp,     21.0_wp,    0.95_wp,  & !  4 z0: Same as asphalt, Estimation, eps: Mandanici et al. (2016)
                                              5.0E-2_wp, 5.0E-4_wp,     22.0_wp,    0.93_wp,  & !  5 z0: Same as asphalt, eps: Göttsche & Hulley (2012)
                                              5.0E-2_wp, 5.0E-4_wp,     23.0_wp,    0.94_wp,  & !  6 z0: Same as asphalt, eps: Mandanici et al. (2016)
                                              5.0E-2_wp, 5.0E-4_wp,     24.0_wp,    0.70_wp,  & !  7 z0: Same as asphalt, eps: Mandanici et al. (2016)
                                              5.0E-2_wp, 5.0E-4_wp,     25.0_wp,    0.90_wp,  & !  8 z0: Same as asphalt, eps: Oke (1987)
                                              5.0E-2_wp, 5.0E-4_wp,     26.0_wp,    0.92_wp,  & !  9 z0: Same as asphalt, eps: Göttsche & Hulley (2012)
                                              5.0E-2_wp, 5.0E-4_wp,     27.0_wp,    0.92_wp,  & ! 10 z0: Same as asphalt, eps: Same as gravel
                                              5.0E-2_wp, 5.0E-4_wp,     28.0_wp,    0.95_wp,  & ! 11 z0: Same as asphalt, eps: AHLRBORN AMiR documentation
                                              5.0E-2_wp, 5.0E-4_wp,     29.0_wp,    0.88_wp,  & ! 12 z0: Same as asphalt, eps: Estimation
                                              5.0E-2_wp, 5.0E-4_wp,     30.0_wp,    0.91_wp,  & ! 13 z0: Same as asphalt, eps: Heldens (2010)
                                              5.0E-2_wp, 5.0E-4_wp,     31.0_wp,    0.95_wp,  & ! 14 z0: Same as asphalt, eps: Gustin et al. (2018)
                                              5.0E-2_wp, 5.0E-4_wp,     32.0_wp,    0.91_wp   & ! 15 z0: Same as asphalt, eps: Engineering toolbox
                                                                          /), (/ 4, 15 /) )
!
!-- Pavement subsurface parameters part 1: thermal conductivity (W/m/K)
!--   0.0-0.01, 0.01-0.03, 0.03-0.07,    0.07-0.15,    0.15-0.30,    0.30-0.50,    0.50-1.25,    1.25-3.00
    REAL(wp), DIMENSION(0:7,1:15), PARAMETER :: pavement_subsurface_pars_1 = RESHAPE( (/                    &
       1.00_wp,   1.00_wp,   1.00_wp,      2.10_wp,      2.10_wp,      0.40_wp,      0.40_wp, 9999999.9_wp, & !  1 Mix of asphalt and concrete
       0.82_wp,   0.82_wp,   0.82_wp,      2.10_wp,      2.10_wp,      0.40_wp,      0.40_wp, 9999999.9_wp, & !  2 Masson et al. (2002)
       1.51_wp,   1.51_wp,   1.51_wp,      2.10_wp,      2.10_wp,      0.40_wp,      0.40_wp, 9999999.9_wp, & !  3 Masson et al. (2002) + Oke (1987)
       2.19_wp,   2.19_wp,   2.19_wp,      2.19_wp,      2.10_wp,      0.40_wp,      0.40_wp, 9999999.9_wp, & !  4 Masson et al. (2002) + Oke (1987)
       2.19_wp,   2.19_wp,   2.19_wp,      2.19_wp,      2.10_wp,      0.40_wp,      0.40_wp, 9999999.9_wp, & !  5 Same as sett
       2.19_wp,   2.19_wp,   2.19_wp,      2.19_wp,      2.10_wp,      0.40_wp,      0.40_wp, 9999999.9_wp, & !  6 Same as sett
       53.3_wp,   53.3_wp,   53.3_wp,      53.3_wp,      53.3_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, & !  7 Oke (1987)
       0.12_wp,   0.12_wp,   0.12_wp,      0.12_wp,      0.12_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, & !  8 Yaghoobian et al. (2009)
       1.40_wp,   1.40_wp,   1.40_wp,      1.40_wp,      1.40_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, & !  9 Masson et al. (2002)
       1.40_wp,   1.40_wp,   1.40_wp,      1.40_wp,      1.40_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, & ! 10 Same as gravel
       1.40_wp,   1.40_wp,   1.40_wp,      1.40_wp,      1.40_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, & ! 11 Same as gravel
       0.20_wp,   0.20_wp,   0.20_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, & ! 12 Skogsberg & Lunberg (2005)
       0.12_wp,   0.12_wp,   0.12_wp,      0.12_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, & ! 13 Heldens (2010)
       0.42_wp,   0.42_wp,   0.42_wp,      0.42_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, & ! 14 Yaghoobian et al. (2009)
       0.84_wp,   0.84_wp,   0.84_wp,      0.84_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp  & ! 15 Oke (1987)
       /), (/ 8, 15 /) )

!
!-- Pavement subsurface parameters part 2: volumetric heat capacity (J/m3/K)
!--     0.0-0.01, 0.01-0.03, 0.03-0.07, 0.07-0.15,       0.15-0.30,    0.30-0.50,    0.50-1.25,    1.25-3.00
    REAL(wp), DIMENSION(0:7,1:15), PARAMETER :: pavement_subsurface_pars_2 = RESHAPE( (/                      &
       2.00E6_wp, 2.00E6_wp, 2.00E6_wp,    2.00E6_wp,    2.00E6_wp,    1.40E6_wp,    1.40E6_wp, 9999999.9_wp, & !  1 Mix of asphalt and concrete
       1.94E6_wp, 1.94E6_wp, 1.94E6_wp,    2.00E6_wp,    2.00E6_wp,    1.40E6_wp,    1.40E6_wp, 9999999.9_wp, & !  2 Masson et al. (2002) + Oke (1987)
       2.11E6_wp, 2.11E6_wp, 2.11E6_wp,    2.00E6_wp,    2.00E6_wp,    1.40E6_wp,    1.40E6_wp, 9999999.9_wp, & !  3 Masson et al. (2002) + Oke (1987)
       2.25E6_wp, 2.25E6_wp, 2.25E6_wp,    2.00E6_wp,    2.00E6_wp,    1.40E6_wp,    1.40E6_wp, 9999999.9_wp, & !  4 Same as paving stones
       2.25E6_wp, 2.25E6_wp, 2.25E6_wp,    2.00E6_wp,    2.00E6_wp,    1.40E6_wp,    1.40E6_wp, 9999999.9_wp, & !  5 Masson et al. (2002) + Oke (1987)
       2.25E6_wp, 2.25E6_wp, 2.25E6_wp,    2.00E6_wp,    2.00E6_wp,    1.40E6_wp,    1.40E6_wp, 9999999.9_wp, & !  6 Same as paving stones
       3.93E6_wp, 3.93E6_wp, 3.93E6_wp,    3.93E6_wp,    3.93E6_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, & !  7 Oke (1987)
       0.66E6_wp, 0.66E6_wp, 0.66E6_wp,    0.66E6_wp,    0.66E6_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, & !  8 Yaghoobian et al. (2009)
       1.76E6_wp, 1.76E6_wp, 1.76E6_wp,    1.76E6_wp,    1.76E6_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, & !  9 Masson et al. (2002)
       1.76E6_wp, 1.76E6_wp, 1.76E6_wp,    1.76E6_wp,    1.76E6_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, & ! 10 Same as gravel
       1.76E6_wp, 1.76E6_wp, 1.76E6_wp,    1.76E6_wp,    1.76E6_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, & ! 11 Same as gravel
       0.50E6_wp, 0.50E6_wp, 0.50E6_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, & ! 12 Sof'fall Inc.
       2.12E6_wp, 2.12E6_wp, 2.12E6_wp,    2.12E6_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, & ! 13 Heldens (2010)
       0.63E6_wp, 0.63E6_wp, 0.63E6_wp,    0.63E6_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, & ! 14 Yaghoobian et al. (2009)
       1.77E6_wp, 1.77E6_wp, 1.77E6_wp,    1.77E6_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp, 9999999.9_wp  & ! 15 Oke (1987)
       /), (/ 8, 15 /) )

!
!-- TO BE FILLED
!-- Water parameters  temperature,      z0,      z0h, lambda_s_s, lambda_s_u, albedo_type, emissivity,
    REAL(wp), DIMENSION(0:6,1:5), PARAMETER :: water_pars = RESHAPE( (/                                   &
                         283.0_wp, 0.001_wp, 0.00001_wp,  1.0E10_wp,  1.0E10_wp,      1.0_wp,    0.95_wp, & ! 1 eps: Engineering toolbox
                         283.0_wp, 0.003_wp, 0.00003_wp,  1.0E10_wp,  1.0E10_wp,      1.0_wp,    0.95_wp, & ! 2 eps: Engineering toolbox
                         283.0_wp, 0.001_wp, 0.00001_wp,  1.0E10_wp,  1.0E10_wp,      1.0_wp,    0.95_wp, & ! 3 eps: Engineering toolbox
                         283.0_wp, 0.001_wp, 0.00001_wp,  1.0E10_wp,  1.0E10_wp,      1.0_wp,    0.95_wp, & ! 4 eps: Engineering toolbox
                         283.0_wp,  0.01_wp,   0.001_wp,  1.0E10_wp,  1.0E10_wp,      1.0_wp,    0.95_wp  & ! 5 eps: Engineering toolbox
                                                                      /), (/ 7, 5 /) )

    SAVE


    PRIVATE


!
!-- Public functions
    PUBLIC lsm_boundary_condition,                                                                 &
           lsm_check_data_output,                                                                  &
           lsm_check_data_output_pr,                                                               &
           lsm_check_data_output_ts,                                                               &
           lsm_check_parameters,                                                                   &
           lsm_define_netcdf_grid,                                                                 &
           lsm_data_output_2d,                                                                     &
           lsm_data_output_3d,                                                                     &
           lsm_energy_balance,                                                                     &
           lsm_header,                                                                             &
           lsm_init,                                                                               &
           lsm_init_arrays,                                                                        &
           lsm_parin,                                                                              &
           lsm_soil_model,                                                                         &
           lsm_swap_timelevel,                                                                     &
           lsm_statistics,                                                                         &
           lsm_rrd_local,                                                                          &
           lsm_timestep,                                                                           &
           lsm_vm_sampling,                                                                        &
           lsm_wrd_local,                                                                          &
           lsm_3d_data_averaging
!
!-- Public parameters, constants and initial values
    PUBLIC aero_resist_kray,                                                                       &
           dt_lsm,                                                                                 &
           skip_time_do_lsm

!
!-- Public grid variables
    PUBLIC nzb_soil, nzs, nzt_soil, zs

!
!-- Public prognostic variables
    PUBLIC m_soil,                                                                                 &
           t_soil

!
!-- Public variable for biogenic module.
    PUBLIC soil_moisture


    INTERFACE lsm_boundary_condition
       MODULE PROCEDURE lsm_boundary_condition
    END INTERFACE lsm_boundary_condition

    INTERFACE lsm_check_data_output
       MODULE PROCEDURE lsm_check_data_output
    END INTERFACE lsm_check_data_output

    INTERFACE lsm_check_data_output_pr
       MODULE PROCEDURE lsm_check_data_output_pr
    END INTERFACE lsm_check_data_output_pr

    INTERFACE lsm_check_data_output_ts
       MODULE PROCEDURE lsm_check_data_output_ts
    END INTERFACE lsm_check_data_output_ts

    INTERFACE lsm_check_parameters
       MODULE PROCEDURE lsm_check_parameters
    END INTERFACE lsm_check_parameters

    INTERFACE lsm_3d_data_averaging
       MODULE PROCEDURE lsm_3d_data_averaging
    END INTERFACE lsm_3d_data_averaging

    INTERFACE lsm_data_output_2d
       MODULE PROCEDURE lsm_data_output_2d
    END INTERFACE lsm_data_output_2d

    INTERFACE lsm_data_output_3d
       MODULE PROCEDURE lsm_data_output_3d
    END INTERFACE lsm_data_output_3d

    INTERFACE lsm_define_netcdf_grid
       MODULE PROCEDURE lsm_define_netcdf_grid
    END INTERFACE lsm_define_netcdf_grid

    INTERFACE lsm_energy_balance
       MODULE PROCEDURE lsm_energy_balance
    END INTERFACE lsm_energy_balance

    INTERFACE lsm_header
       MODULE PROCEDURE lsm_header
    END INTERFACE lsm_header

    INTERFACE lsm_init
       MODULE PROCEDURE lsm_init
    END INTERFACE lsm_init

    INTERFACE lsm_init_arrays
       MODULE PROCEDURE lsm_init_arrays
    END INTERFACE lsm_init_arrays

    INTERFACE lsm_parin
       MODULE PROCEDURE lsm_parin
    END INTERFACE lsm_parin

    INTERFACE lsm_soil_model
       MODULE PROCEDURE lsm_soil_model
    END INTERFACE lsm_soil_model

    INTERFACE lsm_statistics
       MODULE PROCEDURE lsm_statistics
    END INTERFACE lsm_statistics

    INTERFACE lsm_swap_timelevel
       MODULE PROCEDURE lsm_swap_timelevel
    END INTERFACE lsm_swap_timelevel

    INTERFACE lsm_rrd_local
       MODULE PROCEDURE lsm_rrd_local_ftn
       MODULE PROCEDURE lsm_rrd_local_mpi
    END INTERFACE lsm_rrd_local

    INTERFACE lsm_timestep
       MODULE PROCEDURE lsm_timestep
    END INTERFACE lsm_timestep

    INTERFACE lsm_vm_sampling
       MODULE PROCEDURE lsm_vm_sampling
    END INTERFACE lsm_vm_sampling

    INTERFACE lsm_wrd_local
       MODULE PROCEDURE lsm_wrd_local
    END INTERFACE lsm_wrd_local

 CONTAINS


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Set internal Neumann boundary condition at outer soil grid points for temperature and humidity.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_boundary_condition

    IMPLICIT NONE

    INTEGER(iwp) :: i      !< grid index x-direction
    INTEGER(iwp) :: ioff   !< offset index x-direction indicating location of soil grid point
    INTEGER(iwp) :: j      !< grid index y-direction
    INTEGER(iwp) :: joff   !< offset index x-direction indicating location of soil grid point
    INTEGER(iwp) :: k      !< grid index z-direction
    INTEGER(iwp) :: koff   !< offset index x-direction indicating location of soil grid point
    INTEGER(iwp) :: m      !< running index surface elements

    DO  m = 1, surf_lsm%ns
       ioff = surf_lsm%ioff(m)
       joff = surf_lsm%joff(m)
       koff = surf_lsm%koff(m)
       i = surf_lsm%i(m)
       j = surf_lsm%j(m)
       k = surf_lsm%k(m)
       pt(k+koff,j+joff,i+ioff) = pt(k,j,i)
    ENDDO
!
!-- In case of humidity, set boundary conditions also for q and vpt.
    IF ( humidity )  THEN
       DO  m = 1, surf_lsm%ns
          ioff = surf_lsm%ioff(m)
          joff = surf_lsm%joff(m)
          koff = surf_lsm%koff(m)
          i = surf_lsm%i(m)
          j = surf_lsm%j(m)
          k = surf_lsm%k(m)
          q(k+koff,j+joff,i+ioff)   = q(k,j,i)
          vpt(k+koff,j+joff,i+ioff) = vpt(k,j,i)
       ENDDO
    ENDIF

 END SUBROUTINE lsm_boundary_condition

!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Check data output for land surface model
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_check_data_output( var, unit, i, ilen, k )


    USE control_parameters,                                                                        &
        ONLY:  data_output,                                                                        &
               message_string

    IMPLICIT NONE

    CHARACTER (LEN=*) ::  unit  !<
    CHARACTER (LEN=*) ::  var   !<

    INTEGER(iwp) :: i
    INTEGER(iwp) :: ilen
    INTEGER(iwp) :: k

    SELECT CASE ( TRIM( var ) )

       CASE ( 'm_soil' )
          IF (  .NOT.  land_surface )  THEN
             message_string = 'output of "' // TRIM( var ) // '" requi' //                         &
                              'res the land surface model to be switched on'
             CALL message( 'lsm_check_data_output', 'LSM0001', 1, 2, 0, 6, 0 )
          ENDIF
          unit = 'm3/m3'

       CASE ( 't_soil' )
          IF (  .NOT.  land_surface )  THEN
             message_string = 'output of "' // TRIM( var ) // '" requi' //                         &
                              'res the land surface model to be switched on'
             CALL message( 'lsm_check_data_output', 'LSM0001', 1, 2, 0, 6, 0 )
          ENDIF
          unit = 'K'

       CASE ( 'lai*', 'c_liq*', 'c_soil*', 'c_veg*', 'm_liq*', 'qsws_liq*', 'qsws_soil*',          &
              'qsws_veg*', 'r_s*' )
          IF ( k == 0  .OR.  data_output(i)(ilen-2:ilen) /= '_xy' )  THEN
             message_string = 'illegal value for data_output: "' //                                &
                              TRIM( var ) // '" & only 2d-horizontal ' //                          &
                              'cross sections are allowed for this value'
             CALL message( 'lsm_check_data_output', 'LSM0002', 1, 2, 0, 6, 0 )
          ENDIF
          IF ( TRIM( var ) == 'lai*'  .AND.  .NOT. land_surface )  THEN
             message_string = 'output of "' // TRIM( var ) // '" requi' //                         &
                              'res the land surface model to be switched on'
             CALL message( 'lsm_check_data_output', 'LSM0003', 1, 2, 0, 6, 0 )
          ENDIF
          IF ( TRIM( var ) == 'c_liq*'  .AND.  .NOT. land_surface )  THEN
             message_string = 'output of "' // TRIM( var ) // '" requi' //                         &
                              'res the land surface model to be switched on'
             CALL message( 'lsm_check_data_output', 'LSM0003', 1, 2, 0, 6, 0 )
          ENDIF
          IF ( TRIM( var ) == 'c_soil*'  .AND.  .NOT. land_surface )  THEN
             message_string = 'output of "' // TRIM( var ) // '" requi' //                         &
                              'res the land surface model to be switched on'
             CALL message( 'lsm_check_data_output', 'LSM0003', 1, 2, 0, 6, 0 )
          ENDIF
          IF ( TRIM( var ) == 'c_veg*'  .AND.  .NOT. land_surface )  THEN
             message_string = 'output of "' // TRIM( var ) // '" requi' //                         &
                              'res the land surface model to be switched on'
             CALL message( 'lsm_check_data_output', 'LSM0003', 1, 2, 0, 6, 0 )
          ENDIF
          IF ( TRIM( var ) == 'm_liq*'  .AND.  .NOT. land_surface )  THEN
             message_string = 'output of "' // TRIM( var ) // '" requi' //                         &
                              'res the land surface model to be switched on'
             CALL message( 'lsm_check_data_output', 'LSM0003', 1, 2, 0, 6, 0 )
          ENDIF
          IF ( TRIM( var ) == 'qsws_liq*'  .AND.  .NOT. land_surface )  THEN
             message_string = 'output of "' // TRIM( var ) // '" requi' //                         &
                              'res the land surface model to be switched on'
             CALL message( 'lsm_check_data_output', 'LSM0003', 1, 2, 0, 6, 0 )
          ENDIF
          IF ( TRIM( var ) == 'qsws_soil*'  .AND.  .NOT. land_surface )  THEN
             message_string = 'output of "' // TRIM( var ) // '" requi' //                         &
                              'res the land surface model to be switched on'
             CALL message( 'lsm_check_data_output', 'LSM0003', 1, 2, 0, 6, 0 )
          ENDIF
          IF ( TRIM( var ) == 'qsws_veg*'  .AND.  .NOT. land_surface )  THEN
             message_string = 'output of "' // TRIM( var ) // '" requi' //                         &
                              'res the land surface model to be switched on'
             CALL message( 'lsm_check_data_output', 'LSM0003', 1, 2, 0, 6, 0 )
          ENDIF
          IF ( TRIM( var ) == 'r_s*'  .AND.  .NOT. land_surface )  THEN
             message_string = 'output of "' // TRIM( var ) // '" requi' //                         &
                              'res the land surface model to be switched on'
             CALL message( 'lsm_check_data_output', 'LSM0003', 1, 2, 0, 6, 0 )
          ENDIF

          IF ( TRIM( var ) == 'lai*'   )      unit = 'none'
          IF ( TRIM( var ) == 'c_liq*' )      unit = 'none'
          IF ( TRIM( var ) == 'c_soil*')      unit = 'none'
          IF ( TRIM( var ) == 'c_veg*' )      unit = 'none'
          IF ( TRIM( var ) == 'm_liq*'     )  unit = 'm'
          IF ( TRIM( var ) == 'qsws_liq*'  )  unit = 'W/m2'
          IF ( TRIM( var ) == 'qsws_soil*' )  unit = 'W/m2'
          IF ( TRIM( var ) == 'qsws_veg*'  )  unit = 'W/m2'
          IF ( TRIM( var ) == 'r_s*')         unit = 's/m'

       CASE DEFAULT
          unit = 'illegal'

    END SELECT


 END SUBROUTINE lsm_check_data_output



!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Check data output of profiles for land surface model
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_check_data_output_pr( variable, var_count, unit, dopr_unit )

    USE control_parameters,                                                                        &
        ONLY:  data_output_pr,                                                                     &
               message_string

    USE indices

    USE profil_parameter

    USE statistics

    IMPLICIT NONE

    CHARACTER (LEN=*) ::  dopr_unit !< local value of dopr_unit
    CHARACTER (LEN=*) ::  unit      !<
    CHARACTER (LEN=*) ::  variable  !<

    INTEGER(iwp) ::  var_count     !<

    SELECT CASE ( TRIM( variable ) )

       CASE ( 't_soil', '#t_soil' )
          IF (  .NOT.  land_surface )  THEN
             message_string = 'data_output_pr = ' // TRIM( data_output_pr(var_count) ) // ' is ' //&
                              'only implemented for the land surface model'
             CALL message( 'lsm_check_data_output_pr', 'LSM0004', 1, 2, 0, 6, 0 )
          ELSE
             dopr_index(var_count) = 89
             dopr_unit     = 'K'
             hom(0:nzs-1,2,89,:)  = SPREAD( - zs(nzb_soil:nzt_soil), 2, statistic_regions+1 )
             IF ( data_output_pr(var_count)(1:1) == '#' )  THEN
                dopr_initial_index(var_count) = 90
                hom(0:nzs-1,2,90,:)   = SPREAD( - zs(nzb_soil:nzt_soil), 2, statistic_regions+1 )
                data_output_pr(var_count)     = data_output_pr(var_count)(2:)
             ENDIF
             unit = dopr_unit
          ENDIF

       CASE ( 'm_soil', '#m_soil' )
          IF (  .NOT.  land_surface )  THEN
             message_string = 'data_output_pr = ' // TRIM( data_output_pr(var_count) ) // ' is ' //&
                              'only implemented for the land surface model'
             CALL message( 'lsm_check_data_output_pr', 'LSM0004', 1, 2, 0, 6, 0 )
          ELSE
             dopr_index(var_count) = 91
             dopr_unit     = 'm3/m3'
             hom(0:nzs-1,2,91,:)  = SPREAD( - zs(nzb_soil:nzt_soil), 2, statistic_regions+1 )
             IF ( data_output_pr(var_count)(1:1) == '#' )  THEN
                dopr_initial_index(var_count) = 92
                hom(0:nzs-1,2,92,:)   = SPREAD( - zs(nzb_soil:nzt_soil), 2, statistic_regions+1 )
                data_output_pr(var_count)     = data_output_pr(var_count)(2:)
             ENDIF
             unit = dopr_unit
          ENDIF


       CASE DEFAULT
          unit = 'illegal'

    END SELECT


 END SUBROUTINE lsm_check_data_output_pr


 !--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Check data output of timeseries for bulk cloud module
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_check_data_output_ts( dots_max, dots_num, dots_label, dots_unit )

    INTEGER(iwp),      INTENT(IN)     ::  dots_max  !<
    INTEGER(iwp),      INTENT(INOUT)  ::  dots_num  !<

    CHARACTER(LEN=*), DIMENSION(dots_max), INTENT(INOUT)  ::  dots_label  !<
    CHARACTER(LEN=*), DIMENSION(dots_max), INTENT(INOUT)  ::  dots_unit   !<


!
!-- LSM time series:
!-- For each time series quantity you have to give a label and a unit, which will be used for the
!-- NetCDF file. The value of dots_num has to be increased by the number of new time series
!-- quantities. The start index for LSM time series is stored in dots_start_index_lsm and later used
!-- to address the module specific time series values.
    dots_start_index_lsm = dots_num + 1

    dots_num = dots_num + 1
    dots_label(dots_num) = 'ghf'
    dots_unit(dots_num)  = 'W/m2'

    dots_num = dots_num + 1
    dots_label(dots_num) = 'qsws_liq'
    dots_unit(dots_num)  = 'W/m2'

    dots_num = dots_num + 1
    dots_label(dots_num) = 'qsws_soil'
    dots_unit(dots_num)  = 'W/m2'

    dots_num = dots_num + 1
    dots_label(dots_num) = 'qsws_veg'
    dots_unit(dots_num)  = 'W/m2'

    dots_num = dots_num + 1
    dots_label(dots_num) = 'r_a'
    dots_unit(dots_num)  = 's/m'

    dots_num = dots_num + 1
    dots_label(dots_num) = 'r_s'
    dots_unit(dots_num)  = 's/m'

 END SUBROUTINE lsm_check_data_output_ts


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Check parameters routine for land surface model
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_check_parameters

    USE control_parameters,                                                                        &
        ONLY:  bc_pt_b,                                                                            &
               bc_q_b,                                                                             &
               constant_flux_layer,                                                                &
               message_string

    IMPLICIT NONE

    INTEGER(iwp) ::  i        !< running index, x-dimension
    INTEGER(iwp) ::  j        !< running index, y-dimension
    INTEGER(iwp) ::  k        !< running index, z-dimension

    LOGICAL      ::  dynamic_soil_input_parent !< flag indicating the presence of a dynamic input file for the parent

!
!-- Check for a valid setting of surface_type. The default value is 'read_from_file'.
!-- In that case, the surface types are read from the static input file.
    IF ( TRIM( surface_type ) /= 'vegetation'  .AND.  TRIM( surface_type ) /= 'pavement'  .AND.    &
         TRIM( surface_type ) /= 'water'       .AND.  TRIM( surface_type ) /= 'netcdf'  .AND.      &
         TRIM( surface_type ) /= 'read_from_file' )  THEN
       message_string = 'unknown surface type: surface_type = "' // TRIM( surface_type ) // '"'
       CALL message( 'lsm_check_parameters', 'LSM0005', 1, 2, 0, 6, 0 )
    ENDIF
!
!-- Give informative message about deprecated setting surface_type.
    IF ( TRIM( surface_type ) == 'netcdf' )  THEN
       message_string = 'surface_type = "netcdf" is deprecated and will be removed. ' //           &
                        'Please use "read_from_file" instead.'
       CALL message( 'lsm_check_parameters', 'LSM0006', 0, 0, 0, 0, 0 )
    ENDIF
!
!-- Dirichlet boundary conditions are required as the surface fluxes are calculated from the
!-- temperature/humidity gradients in the land surface model.
    IF ( bc_pt_b == 'neumann'  .OR.  bc_q_b == 'neumann' )  THEN
       message_string = 'lsm requires setting of' // 'bc_pt_b = "dirichlet" and ' //               &
                        'bc_q_b  = "dirichlet"'
       CALL message( 'lsm_check_parameters', 'LSM0007', 1, 2, 0, 6, 0 )
    ENDIF

    IF ( .NOT. constant_flux_layer )  THEN
       message_string = 'lsm requires constant_flux_layer = .T.'
       CALL message( 'lsm_check_parameters', 'LSM0008', 1, 2, 0, 6, 0 )
    ENDIF

    IF ( .NOT. radiation )  THEN
       message_string = 'lsm requires the radiation model to be switched on'
       CALL message( 'lsm_check_parameters', 'LSM0009', 1, 2, 0, 6, 0 )
    ENDIF
!
!-- Check if soil types are set within a valid range.
    IF ( TRIM( surface_type ) == 'vegetation'  .OR.  TRIM( surface_type ) == 'pavement' )  THEN
       IF ( soil_type < LBOUND( soil_pars, 2 )  .AND.  soil_type > UBOUND( soil_pars, 2 ) )  THEN
          WRITE( message_string, * ) 'soil_type = ', soil_type, ' is out of the valid range ' //   &
                                     '(surface type "vegetation" or "pavement")'
          CALL message( 'lsm_check_parameters', 'LSM0010', 2, 2, 0, 6, 0 )
       ENDIF
    ENDIF
    IF ( TRIM( surface_type ) == 'netcdf'  .OR.  TRIM( surface_type ) == 'read_from_file' )  THEN
       IF ( soil_type_f%from_file )  THEN
          DO  i = nxl, nxr
             DO  j = nys, nyn
                IF ( soil_type_f%var_2d(j,i) /= soil_type_f%fill  .AND.                            &
                     ( soil_type_f%var_2d(j,i) < LBOUND( soil_pars, 2 ) .OR.                       &
                       soil_type_f%var_2d(j,i) > UBOUND( soil_pars, 2 ) ) )                        &
                THEN
                   WRITE( message_string, * ) 'soil_type = ', soil_type_f%var_2d(j,i),             &
                                              ' is out of the valid range at (j,i) = ', j, i,      &
                                              ' (surface type "netcdf" or "read_from_file")'
                   CALL message( 'lsm_check_parameters', 'LSM0011', 2, 2, myid, 6, 0 )
                ENDIF
             ENDDO
          ENDDO
       ENDIF
    ENDIF
!
!-- Check if vegetation types are set within a valid range.
    IF ( TRIM( surface_type ) == 'vegetation' )  THEN
       IF ( vegetation_type < LBOUND( vegetation_pars, 2 )  .AND.                                  &
            vegetation_type > UBOUND( vegetation_pars, 2 ) )  THEN
          WRITE( message_string, * ) 'vegetation_type = ', vegetation_type,                        &
                                     ' is out of the valid range (surface type "vegetation")'
          CALL message( 'lsm_check_parameters', 'LSM0012', 2, 2, 0, 6, 0 )
       ENDIF
    ENDIF
    IF ( TRIM( surface_type ) == 'netcdf'  .OR.  TRIM( surface_type ) == 'read_from_file' )  THEN
       IF ( vegetation_type_f%from_file )  THEN
          DO  i = nxl, nxr
             DO  j = nys, nyn
                IF ( vegetation_type_f%var(j,i) /= vegetation_type_f%fill  .AND.                   &
                     ( vegetation_type_f%var(j,i) < LBOUND( vegetation_pars, 2 ) .OR.              &
                       vegetation_type_f%var(j,i) > UBOUND( vegetation_pars, 2 ) ) )               &
                THEN
                   WRITE( message_string, * ) 'vegetation_type = ', vegetation_type_f%var(j,i),    &
                                              ' is out of the valid range at (j,i) = ', j, i,      &
                                              ' (surface type "netcdf" or "read_from_file")'
                   CALL message( 'lsm_check_parameters', 'LSM0013', 2, 2, myid, 6, 0 )
                ENDIF
             ENDDO
          ENDDO
       ENDIF
    ENDIF
!
!-- Check if pavement types are set within a valid range.
    IF ( TRIM( surface_type ) == 'pavement' )  THEN
       IF ( pavement_type < LBOUND( pavement_pars, 2 )  .AND.                                      &
            pavement_type > UBOUND( pavement_pars, 2 ) )  THEN
          WRITE( message_string, * ) 'pavement_type = ', pavement_type,                            &
                                     ' is out of the valid range (surface type "pavement")'
          CALL message( 'lsm_check_parameters', 'LSM0014', 2, 2, 0, 6, 0 )
       ENDIF
    ENDIF
    IF ( TRIM( surface_type ) == 'netcdf'  .OR.  TRIM( surface_type ) == 'read_from_file')  THEN
       IF ( pavement_type_f%from_file )  THEN
          DO  i = nxl, nxr
             DO  j = nys, nyn
                IF ( pavement_type_f%var(j,i) /= pavement_type_f%fill  .AND.                       &
                     ( pavement_type_f%var(j,i) < LBOUND( pavement_pars, 2 ) .OR.                  &
                       pavement_type_f%var(j,i) > UBOUND( pavement_pars, 2 ) ) )                   &
                THEN
                   WRITE( message_string, * ) 'pavement_type = ', pavement_type_f%var(j,i),        &
                                              ' is out of the valid range at (j,i) = ', j, i,      &
                                              ' (surface type "netcdf" or "read_from_file")'
                   CALL message( 'lsm_check_parameters', 'LSM0015', 2, 2, myid, 6, 0 )
                ENDIF
             ENDDO
          ENDDO
       ENDIF
    ENDIF
!
!-- Check if water types are set within a valid range.
    IF ( TRIM( surface_type ) == 'water' )  THEN
       IF ( water_type < LBOUND( water_pars, 2 )  .AND.  water_type > UBOUND( water_pars, 2 ) )    &
       THEN
          WRITE( message_string, * ) 'water_type = ', water_type,                                  &
                                     ' is out of the valid range (surface type "water")'
          CALL message( 'lsm_check_parameters', 'LSM0016', 2, 2, 0, 6, 0 )
       ENDIF
    ENDIF
    IF ( TRIM( surface_type ) == 'netcdf'  .OR.  TRIM( surface_type ) == 'read_from_file' )  THEN
       IF ( water_type_f%from_file )  THEN
          DO  i = nxl, nxr
             DO  j = nys, nyn
                IF ( water_type_f%var(j,i) /= water_type_f%fill  .AND.                             &
                     ( water_type_f%var(j,i) < LBOUND( water_pars, 2 ) .OR.                        &
                       water_type_f%var(j,i) > UBOUND( water_pars, 2 ) ) )                         &
                THEN
                   WRITE( message_string, * ) 'water_type = ', water_type_f%var(j,i),              &
                                              ' is out  of the valid range at (j,i) = ', j, i,     &
                                              ' (surface type "netcdf" or "read_from_file")'
                   CALL message( 'lsm_check_parameters', 'LSM0017', 2, 2, myid, 6, 0 )
                ENDIF
             ENDDO
          ENDDO
       ENDIF
    ENDIF
!
!-- Check further settings for consistency.
    IF ( TRIM( surface_type ) == 'vegetation' )  THEN

       IF ( vegetation_type == 0 )  THEN
          IF ( min_canopy_resistance == 9999999.9_wp )  THEN
             message_string = 'vegetation_type = 0 (user defined)' //                              &
                              ' requires setting of min_canopy_resistance' // '/= 9999999.9'
             CALL message( 'lsm_check_parameters', 'LSM0018', 1, 2, 0, 6, 0 )
          ENDIF

          IF ( leaf_area_index == 9999999.9_wp )  THEN
             message_string = 'vegetation_type = 0 (user_defined)' //                              &
                              ' requires setting of leaf_area_index' // '/= 9999999.9'
             CALL message( 'lsm_check_parameters', 'LSM0018', 1, 2, 0, 6, 0 )
          ENDIF

          IF ( vegetation_coverage == 9999999.9_wp )  THEN
             message_string = 'vegetation_type = 0 (user_defined)' //                              &
                              ' requires setting of vegetation_coverage' // '/= 9999999.9'
             CALL message( 'lsm_check_parameters', 'LSM0018', 1, 2, 0, 6, 0 )
          ENDIF

          IF ( canopy_resistance_coefficient == 9999999.9_wp)  THEN
             message_string = 'vegetation_type = 0 (user_defined)' //                              &
                              ' requires setting of' // 'canopy_resistance_coefficient /= 9999999.9'
             CALL message( 'lsm_check_parameters', 'LSM0018', 1, 2, 0, 6, 0 )
          ENDIF

          IF ( lambda_surface_stable == 9999999.9_wp )  THEN
             message_string = 'vegetation_type = 0 (user_defined)' //                              &
                              ' requires setting of lambda_surface_stable' // '/= 9999999.9'
             CALL message( 'lsm_check_parameters', 'LSM0018', 1, 2, 0, 6, 0 )
          ENDIF

          IF ( lambda_surface_unstable == 9999999.9_wp )  THEN
             message_string = 'vegetation_type = 0 (user_defined)' //                              &
                              ' requires setting of lambda_surface_unstable' // '/= 9999999.9'
             CALL message( 'lsm_check_parameters', 'LSM0018', 1, 2, 0, 6, 0 )
          ENDIF

          IF ( f_shortwave_incoming == 9999999.9_wp )  THEN
             message_string = 'vegetation_type = 0 (user_defined)' //                              &
                              ' requires setting of f_shortwave_incoming' // '/= 9999999.9'
             CALL message( 'lsm_check_parameters', 'LSM0018', 1, 2, 0, 6, 0 )
          ENDIF

          IF ( z0_vegetation == 9999999.9_wp )  THEN
             message_string = 'vegetation_type = 0 (user_defined)' //                              &
                              ' requires setting of z0_vegetation' // '/= 9999999.9'
             CALL message( 'lsm_check_parameters', 'LSM0018', 1, 2, 0, 6, 0 )
          ENDIF

          IF ( z0h_vegetation == 9999999.9_wp )  THEN
             message_string = 'vegetation_type = 0 (user_defined)' //                              &
                              ' requires setting of z0h_vegetation' // '/= 9999999.9'
             CALL message( 'lsm_check_parameters', 'LSM0018', 1, 2, 0, 6, 0 )
          ENDIF
       ENDIF

       IF ( vegetation_type == 1 )  THEN
          IF ( vegetation_coverage /= 9999999.9_wp  .AND.  vegetation_coverage /= 0.0_wp )  THEN
             message_string = 'vegetation_type = 1 (bare soil)' //                                 &
                              ' requires vegetation_coverage = 0'
             CALL message( 'lsm_check_parameters', 'LSM0019', 1, 2, 0, 6, 0 )
          ENDIF
       ENDIF

    ENDIF

    IF ( TRIM( surface_type ) == 'water' )  THEN

       IF ( water_type == 0 )  THEN

          IF ( z0_water == 9999999.9_wp )  THEN
             message_string = 'water_type = 0 (user_defined)' //                                   &
                              ' requires setting of z0_water' // '/= 9999999.9'
             CALL message( 'lsm_check_parameters', 'LSM0020', 1, 2, 0, 6, 0 )
          ENDIF

          IF ( z0h_water == 9999999.9_wp )  THEN
             message_string = 'water_type = 0 (user_defined)' //                                   &
                              ' requires setting of z0h_water' // '/= 9999999.9'
             CALL message( 'lsm_check_parameters', 'LSM0021', 1, 2, 0, 6, 0 )
          ENDIF

          IF ( water_temperature == 9999999.9_wp )  THEN
             message_string = 'water_type = 0 (user_defined)' //                                   &
                              ' requires setting of water_temperature' // '/= 9999999.9'
             CALL message( 'lsm_check_parameters', 'LSM0022', 1, 2, 0, 6, 0 )
          ENDIF

       ENDIF

    ENDIF

    IF ( TRIM( surface_type ) == 'pavement' )  THEN

       IF ( ANY( dz_soil /= 9999999.9_wp )  .AND.  pavement_type /= 0 )  THEN
          message_string = 'non-default setting of dz_soil ' //                                    &
                           'does not allow to use pavement_type /= 0'
          CALL message( 'lsm_check_parameters', 'LSM0023', 1, 2, 0, 6, 0 )
       ENDIF

       IF ( pavement_type == 0 )  THEN

          IF ( z0_pavement == 9999999.9_wp )  THEN
             message_string = 'pavement_type = 0 (user_defined)' //                                &
                              ' requires setting of z0_pavement' // '/= 9999999.9'
             CALL message( 'lsm_check_parameters', 'LSM0024', 1, 2, 0, 6, 0 )
          ENDIF

          IF ( z0h_pavement == 9999999.9_wp )  THEN
             message_string = 'pavement_type = 0 (user_defined)' //                                &
                              ' requires setting of z0h_pavement' // '/= 9999999.9'
             CALL message( 'lsm_check_parameters', 'LSM0025', 1, 2, 0, 6, 0 )
          ENDIF

          IF ( pavement_heat_conduct == 9999999.9_wp )  THEN
             message_string = 'pavement_type = 0 (user_defined)' //                                &
                              ' requires setting of pavement_heat_conduct' // '/= 9999999.9'
             CALL message( 'lsm_check_parameters', 'LSM0026', 1, 2, 0, 6, 0 )
          ENDIF

           IF ( pavement_heat_capacity == 9999999.9_wp )  THEN
             message_string = 'pavement_type = 0 (user_defined)' //                                &
                              ' requires setting of pavement_heat_capacity' // '/= 9999999.9'
             CALL message( 'lsm_check_parameters', 'LSM0027', 1, 2, 0, 6, 0 )
          ENDIF

          IF ( pavement_depth_level == 0 )  THEN
             message_string = 'pavement_type = 0 (user_defined)' //                                &
                              ' requires setting of pavement_depth_level' // '/= 0'
             CALL message( 'lsm_check_parameters', 'LSM0028', 1, 2, 0, 6, 0 )
          ENDIF

       ENDIF

    ENDIF

    IF ( TRIM( surface_type ) == 'netcdf'  .OR.  TRIM( surface_type ) == 'read_from_file' )  THEN
       IF ( pavement_type_f%from_file )  THEN
          IF ( ANY( pavement_type_f%var /= pavement_type_f%fill )  .AND.                           &
               ANY( dz_soil /= 9999999.9_wp ) )  THEN
             message_string = 'pavement-surfaces are not allowed in ' //                           &
                              'combination with a non-default setting of dz_soil'
             CALL message( 'lsm_check_parameters', 'LSM0029', 2, 2, 0, 6, 0 )
          ENDIF
       ENDIF
    ENDIF

!
!-- Temporary message as long as NetCDF input is not available
    IF ( ( TRIM( surface_type ) == 'netcdf'  .OR.  TRIM( surface_type ) == 'read_from_file' )      &
         .AND.  .NOT. input_pids_static )  THEN
       message_string = 'surface_type = "read_from_file" requires a static input file'
       CALL message( 'lsm_check_parameters', 'LSM0030', 1, 2, 0, 6, 0 )
    ENDIF

    IF ( soil_type == 0  .AND.  .NOT. input_pids_static )  THEN

       IF ( alpha_vangenuchten == 9999999.9_wp )  THEN
          message_string = 'soil_type = 0 (user_defined)' //                                       &
                           'requires setting of alpha_vangenuchten' // '/= 9999999.9'
          CALL message( 'lsm_check_parameters', 'LSM0031', 1, 2, 0, 6, 0 )
       ENDIF

       IF ( l_vangenuchten == 9999999.9_wp )  THEN
          message_string = 'soil_type = 0 (user_defined)' //                                       &
                           'requires setting of l_vangenuchten' // '/= 9999999.9'
          CALL message( 'lsm_check_parameters', 'LSM0031', 1, 2, 0, 6, 0 )
       ENDIF

       IF ( n_vangenuchten == 9999999.9_wp )  THEN
          message_string = 'soil_type = 0 (user_defined)' //                                       &
                           'requires setting of n_vangenuchten' // '/= 9999999.9'
          CALL message( 'lsm_check_parameters', 'LSM0031', 1, 2, 0, 6, 0 )
       ENDIF

       IF ( hydraulic_conductivity == 9999999.9_wp )  THEN
          message_string = 'soil_type = 0 (user_defined)' //                                       &
                           'requires setting of hydraulic_conductivity' // '/= 9999999.9'
          CALL message( 'lsm_check_parameters', 'LSM0031', 1, 2, 0, 6, 0 )
       ENDIF

       IF ( saturation_moisture == 9999999.9_wp )  THEN
          message_string = 'soil_type = 0 (user_defined)' //                                       &
                           'requires setting of saturation_moisture' // '/= 9999999.9'
          CALL message( 'lsm_check_parameters', 'LSM0031', 1, 2, 0, 6, 0 )
       ENDIF

       IF ( field_capacity == 9999999.9_wp )  THEN
          message_string = 'soil_type = 0 (user_defined)' //                                       &
                           'requires setting of field_capacity' // '/= 9999999.9'
          CALL message( 'lsm_check_parameters', 'LSM0031', 1, 2, 0, 6, 0 )
       ENDIF

       IF ( wilting_point == 9999999.9_wp )  THEN
          message_string = 'soil_type = 0 (user_defined)' //                                       &
                           'requires setting of wilting_point' // '/= 9999999.9'
          CALL message( 'lsm_check_parameters', 'LSM0031', 1, 2, 0, 6, 0 )
       ENDIF

       IF ( residual_moisture == 9999999.9_wp )  THEN
          message_string = 'soil_type = 0 (user_defined)' //                                       &
                           'requires setting of residual_moisture' // '/= 9999999.9'
          CALL message( 'lsm_check_parameters', 'LSM0031', 1, 2, 0, 6, 0 )
       ENDIF

    ENDIF

!
!-- Determine number of soil layers to be used and check whether an appropriate root fraction is
!-- prescribed.
    nzb_soil = 0
    nzt_soil = -1
    IF ( ALL( dz_soil == 9999999.9_wp ) )  THEN
       nzt_soil = 7
       dz_soil(nzb_soil:nzt_soil) = dz_soil_default
    ELSE
       DO k = 0, 19
          IF ( dz_soil(k) /= 9999999.9_wp )  THEN
             nzt_soil = nzt_soil + 1
          ENDIF
       ENDDO
    ENDIF
    nzs = nzt_soil + 1

!
!-- Check whether valid soil temperatures are prescribed. Only check this if no dynamic soil is not
!-- initialized with dynamic input.
!-- In a nested case, check whether there is a dynamic input file for the child
!-- (input_pids_dynamic = .T.) or one for the parent (inquire without coupling_char).
    INQUIRE( FILE = TRIM( input_file_dynamic ), EXIST = dynamic_soil_input_parent )

    IF ( .NOT. input_pids_dynamic  .AND.  .NOT. dynamic_soil_input_parent )  THEN
       IF ( COUNT( soil_temperature /= 9999999.9_wp ) /= nzs )  THEN
          WRITE( message_string, * ) 'number of soil layers (', nzs, ') does not',                 &
                                     ' match to the number of layers specified',                   &
                                     ' in soil_temperature (',                                     &
                                     COUNT( soil_temperature /= 9999999.9_wp ), ')'
             CALL message( 'lsm_check_parameters', 'LSM0032', 1, 2, 0, 6, 0 )
       ENDIF

       IF ( deep_soil_temperature == 9999999.9_wp )  THEN
             message_string = 'deep_soil_temperature is not set but must be' // '/= 9999999.9'
             CALL message( 'lsm_check_parameters', 'LSM0033', 1, 2, 0, 6, 0 )
       ENDIF
    ENDIF

!
!-- Check whether the sum of all root fractions equals one
    IF ( .NOT. vegetation_type_f%from_file )  THEN
       IF ( vegetation_type == 0 )  THEN
          IF ( SUM( root_fraction(nzb_soil:nzt_soil) ) /= 1.0_wp )  THEN
             message_string = 'vegetation_type = 0 (user_defined)' //                              &
                              ' requires setting of root_fraction' //                              &
                              ' /= 9999999.9 and SUM(root_fraction) = 1'
             CALL message( 'lsm_check_parameters', 'LSM0018', 1, 2, 0, 6, 0 )
          ENDIF
       ENDIF
    ENDIF
!
!-- Check if the static input *_pars variables are correctly dimensioned.
    IF ( pavement_pars_f%from_file )  THEN
       IF ( SIZE( pavement_pars_f%pars ) /= SIZE( pavement_pars, 1 ) )  THEN
          WRITE( message_string, * ) 'dimension size of static input variable pavement_pars is ',  &
                                     SIZE( pavement_pars_f%pars ), ',&',                           &
                                     'but dimension size of ', SIZE( pavement_pars, 1 ),           &
                                     'is required'
          CALL message( 'lsm_check_parameters', 'LSM0034', 2, 2, 0, 6, 0 )
       ENDIF
    ENDIF

    IF ( pavement_subsurface_pars_f%from_file )  THEN
       IF ( SIZE( pavement_subsurface_pars_f%pars ) /= npavement_subsurface_pars )  THEN
          WRITE( message_string, * ) 'dimension size of static input variable ',                   &
                                     'pavement_subsurface_pars is ',                               &
                                     SIZE( pavement_subsurface_pars_f%pars ), ',&',                &
                                     'but dimension size of ', npavement_subsurface_pars,          &
                                     'is required'
          CALL message( 'lsm_check_parameters', 'LSM0035', 2, 2, 0, 6, 0 )
       ENDIF
    ENDIF

    IF ( vegetation_pars_f%from_file )  THEN
       IF ( SIZE( vegetation_pars_f%pars ) /= SIZE( vegetation_pars, 1 ) )  THEN
          WRITE( message_string, * ) 'dimension size of static input variable vegetation_pars is ',&
                                     SIZE( vegetation_pars_f%pars ), '.&',                         &
                                     'Dimension size of ', SIZE( vegetation_pars, 1 ),             &
                                     'is required'
          CALL message( 'lsm_check_parameters', 'LSM0036', 2, 2, 0, 6, 0 )
       ENDIF
    ENDIF

    IF ( water_pars_f%from_file )  THEN
       IF ( SIZE( water_pars_f%pars ) /= SIZE( water_pars, 1 ) )  THEN
          WRITE( message_string, * ) 'dimension size of static input variable water_pars is ',     &
                                     SIZE( water_pars_f%pars ), '.&',                              &
                                     'Dimension size of ', SIZE( water_pars, 1 ), 'is required'
          CALL message( 'lsm_check_parameters', 'LSM0037', 2, 2, 0, 6, 0 )
       ENDIF
    ENDIF
!
!-- Calculate grid spacings. Temperature and moisture are defined at the center of the soil layers,
!-- whereas gradients/fluxes are defined at the edges (_layer)
!-- First, allocate arrays for the soil grid.
    ALLOCATE ( ddz_soil_center(nzb_soil:nzt_soil) )
    ALLOCATE ( ddz_soil(nzb_soil:nzt_soil+1) )
    ALLOCATE ( dz_soil_center(nzb_soil:nzt_soil) )
    ALLOCATE ( zs(nzb_soil:nzt_soil+1) )

    zs(nzb_soil) = 0.5_wp * dz_soil(nzb_soil)
    zs_layer(nzb_soil) = dz_soil(nzb_soil)

    DO  k = nzb_soil+1, nzt_soil
       zs_layer(k) = zs_layer(k-1) + dz_soil(k)
       zs(k) = (zs_layer(k) +  zs_layer(k-1)) * 0.5_wp
    ENDDO

    dz_soil(nzt_soil+1) = zs_layer(nzt_soil) + dz_soil(nzt_soil)
    zs(nzt_soil+1) = zs_layer(nzt_soil) + 0.5_wp * dz_soil(nzt_soil)

    DO  k = nzb_soil, nzt_soil-1
       dz_soil_center(k) = zs(k+1) - zs(k)
       IF ( dz_soil_center(k) <= 0.0_wp )  THEN
          WRITE( message_string, '(A,I5,A)' )  'invalid soil layer configuration found ' //        &
                                               '(dz_soil_center(', k, ') <= 0.0)'
          CALL message( 'lsm_check_parameters', 'LSM0038', 1, 2, 0, 6, 0 )
       ENDIF
    ENDDO

    dz_soil_center(nzt_soil) = zs_layer(k-1) + dz_soil(k) - zs(nzt_soil)

    ddz_soil_center = 1.0_wp / dz_soil_center
    ddz_soil(nzb_soil:nzt_soil) = 1.0_wp / dz_soil(nzb_soil:nzt_soil)

 END SUBROUTINE lsm_check_parameters

!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Calls energy balance solver and soil energy model the surfaces in all directions.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_energy_balance( during_spinup )

    USE control_parameters,                                                                        &
        ONLY:  calc_soil_moisture_during_spinup

    LOGICAL      ::  during_spinup  !< flag indicating soil/wall spinup phase
    LOGICAL      ::  calc_sm        !< flag controlling soil moisture treatment during spinup


    calc_sm = ( .NOT. during_spinup  .OR.  calc_soil_moisture_during_spinup )

    CALL lsm_surface_energy_balance
    CALL lsm_soil_model( calc_sm )

 END SUBROUTINE lsm_energy_balance


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Solver for the energy balance at the surface.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_surface_energy_balance

    USE control_parameters,                                                                        &
        ONLY:  dcep

    USE dcep_mod,                                                                                  &
        ONLY:  fr_urb

    USE pegrid

    USE radiation_model_mod,                                                                       &
        ONLY:  rad_lw_out

    IMPLICIT NONE

    INTEGER(iwp) ::  i      !< running index
    INTEGER(iwp) ::  i_off  !< offset to determine index of surface element, seen from atmospheric grid point, for x
    INTEGER(iwp) ::  j      !< running index
    INTEGER(iwp) ::  j_off  !< offset to determine index of surface element, seen from atmospheric grid point, for y
    INTEGER(iwp) ::  k      !< running index
    INTEGER(iwp) ::  k_off  !< offset to determine index of surface element, seen from atmospheric grid point, for z
    INTEGER(iwp) ::  ks     !< running index
    INTEGER(iwp) ::  m      !< running index concerning wall elements

    LOGICAL ::  horizontal  !< flag indicating horizontal surfaces
    LOGICAL ::  rrtmg_l     !< flag used to replace character statements for rrtmg that prevent vectorization
    LOGICAL ::  runge_l     !< flag used to replace character statements for RK3 that prevent vectorization

    LOGICAL, DIMENSION(1:surf_lsm%ns) ::  force_radiation_call_l_v

    REAL(wp) ::  c_surface_tmp   !< temporary variable for storing the volumetric heat capacity of the surface
    REAL(wp) ::  coef_1          !< coef. for prognostic equation
    REAL(wp) ::  coef_2          !< coef. for prognostic equation
    REAL(wp) ::  dq_s_dt         !< derivate of q_s with respect to T
    REAL(wp) ::  e               !< water vapour pressure
    REAL(wp) ::  e_s_dt          !< derivate of e_s with respect to T
    REAL(wp) ::  f_qsws          !< factor for qsws
    REAL(wp) ::  f_qsws_liq      !< factor for qsws_liq
    REAL(wp) ::  f_qsws_soil     !< factor for qsws_soil
    REAL(wp) ::  f_qsws_veg      !< factor for qsws_veg
    REAL(wp) ::  f_shf           !< factor for shf
    REAL(wp) ::  lambda_soil     !< thermal conductivity of the uppermost soil layer (W/m2/K)
    REAL(wp) ::  lambda_surface  !< Current value of lambda_surface (W/m2/K)
    REAL(wp) ::  m_liq_max       !< maxmimum value of the liq. water reservoir
    REAL(wp) ::  m_min           !< minimum soil moisture
    REAL(wp) ::  tend            !< tendency
    REAL(wp) ::  ueff            !< limited near-surface wind speed - used for calculation of resistance

    REAL(wp), DIMENSION(1:surf_lsm%ns) ::  e_s !< water vapour saturation pressure
    REAL(wp), DIMENSION(1:surf_lsm%ns) ::  f1  !< resistance correction term 1
    REAL(wp), DIMENSION(1:surf_lsm%ns) ::  f2  !< resistance correction term 2
    REAL(wp), DIMENSION(1:surf_lsm%ns) ::  f3  !< resistance correction term 3

    TYPE(surf_type_lsm), POINTER ::  surf_m_liq
    TYPE(surf_type_lsm), POINTER ::  surf_m_liq_p
    TYPE(surf_type_lsm), POINTER ::  surf_t_surface
    TYPE(surf_type_lsm), POINTER ::  surf_t_surface_p
    TYPE(surf_type_lsm), POINTER ::  surf_tm_liq_m
    TYPE(surf_type_lsm), POINTER ::  surf_tt_surface_m

    TYPE(surf_type_lsm), POINTER ::  surf_m_soil
    TYPE(surf_type_lsm), POINTER ::  surf_t_soil

    TYPE(surf_type), POINTER  ::  surf  !< surface-date type variable

    force_radiation_call_l_v = .FALSE.

    IF ( debug_output_timestep )  THEN
       WRITE( debug_string, * ) 'lsm_surface_energy_balance'
       CALL debug_message( debug_string, 'start' )
    ENDIF

    runge_l = ( timestep_scheme(1:5) == 'runge' )
    rrtmg_l = ( radiation_scheme == 'rrtmg' )

    surf              => surf_lsm
    surf_t_surface    => t_surface
    surf_t_surface_p  => t_surface_p
    surf_tt_surface_m => tt_surface_m
    surf_m_liq        => m_liq
    surf_m_liq_p      => m_liq_p
    surf_tm_liq_m     => tm_liq_m
    surf_m_soil       => m_soil
    surf_t_soil       => t_soil
!
!-- Compute aerodynamic resistance.
    !$OMP PARALLEL DO PRIVATE (m, i, j, k, ueff ) SCHEDULE (STATIC)
    DO  m = 1, surf%ns

       i = surf%i(m)
       j = surf%j(m)
       k = surf%k(m)
!
!--    Calculate aerodyamic resistance. At upward facing surfaces, use MOST and the vertical
!--    temperature gradient.
       IF ( surf%upward(m)  .OR.  .NOT. aero_resist_kray )  THEN
!
!--       Dimension of r_a is s/m: K / ( K * m / s )
          surf%r_a(m) = ABS( ( surf%pt1(m) - surf%pt_surface(m) ) /                                &
                             ( surf%ts(m) * surf%us(m) + 1.0E-20_wp ) )
!
!--    At surfaces with other orientation, use the formulation used in the TUF3d model.
!--    (Krayenhoff & Voogt, 2006). Note that this formulation is the equivalent to the
!--    ECMWF formulation using drag coefficients. A roughness length of 0.001 is assumed here for
!--    concrete (the inverse, 1000 is used in the nominator for scaling).
       ELSE
!
!--       Limit wind velocity in order to avoid division by zero.
!--       The nominator can become <= 0.0 for values z0 < 3*10E-4.
          ueff = MAX( SQRT(  ( ( u(k,j,i) + u(k,j,i+1) ) * 0.5_wp )**2 +                           &
                             ( ( v(k,j,i) + v(k,j+1,i) ) * 0.5_wp )**2 +                           &
                             ( ( w(k,j,i) + w(k-1,j,i) ) * 0.5_wp )**2 ),                          &
                      1.0_wp / 4.2_wp * ( 4.0_wp / ( surf%z0(m) * 1000.0_wp ) - 11.8_wp ), 0.1_wp )
!
!--       Dimension of r_a is s/m with denominator in W / ( m2 K )
          surf%r_a(m) = rho_cp / ( surf%z0(m) * 1000.0_wp * ( 11.8_wp + 4.2_wp * ueff ) - 4.0_wp )
       ENDIF

!
!--    Make sure that the resistance does not drop to zero for neutral stratification. Also, set a
!--    maximum resistance to avoid the breakdown of MOST for locations with zero wind speed.
       IF ( surf%r_a(m) <   1.0_wp )  surf%r_a(m) =   1.0_wp
       IF ( surf%r_a(m) > 300.0_wp )  surf%r_a(m) = 300.0_wp
    ENDDO

!
!-- Compute canopy and soil resistance including corresponding correction terms.
    !$OMP PARALLEL DO PRIVATE (e, m, m_min, m_total) SCHEDULE (STATIC)
    DO  m = 1, surf%ns
!
!--    Calculate canopy resistance r_canopy f1-f3 here are defined as 1/f1-f3 as in
!--    ECMWF documentation.
!--    f1: correction for incoming shortwave radiation (stomata close at night).
       f1(m) = MIN( 1.0_wp, ( 0.004_wp * surf%rad_sw_in(m) + 0.05_wp ) /                           &
                                 ( 0.81_wp * ( 0.004_wp * surf%rad_sw_in(m) + 1.0_wp) ) )

!
!--    f2: correction for soil moisture availability to plants (the integrated soil moisture must
!--        thus be considered here)
       m_total = 0.0_wp
       DO  ks = nzb_soil, nzt_soil
           m_total = m_total + surf%root_fr(ks,m) *                                                &
                     MAX( surf_m_soil%var_2d(ks,m), surf%m_wilt(ks,m) )
       ENDDO
!
!--    The calculation of f2 is based on only one wilting point value for all soil layers. The value
!--    at k=nzb_soil is used here as a proxy but might need refinement in the future.
       IF ( m_total > surf%m_wilt(nzb_soil,m)  .AND.  m_total < surf%m_fc(nzb_soil,m) )  THEN
          f2(m) = ( m_total - surf%m_wilt(nzb_soil,m) ) /                                          &
                  ( surf%m_fc(nzb_soil,m) - surf%m_wilt(nzb_soil,m) )
       ELSEIF ( m_total >= surf%m_fc(nzb_soil,m) )  THEN
          f2(m) = 1.0_wp
!
!--    f2 = 0 for very dry soils.
       ELSE
          f2(m) = 1.0E-20_wp
       ENDIF
!
!--    Calculate water vapour pressure at saturation and convert to hPa.
!--    The magnus formula is limited to temperatures up to 333.15 K to avoid negative values of q_s.
       e_s(m) = 0.01_wp * magnus( MIN( surf_t_surface%var_1d(m), 333.15_wp ) )
!
!--    f3: correction for vapour pressure deficit.
       IF ( surf%g_d(m) /= 0.0_wp )  THEN
!
!--       Calculate vapour pressure
          e  = surf%qv1(m) * surface_pressure / ( surf%qv1(m) + rd_d_rv )
          f3(m) = EXP ( - surf%g_d(m) * ( e_s(m) - e ) )
       ELSE
          f3(m) = 1.0_wp
       ENDIF
!
!--    Calculate canopy resistance. In case that c_veg is 0 (bare soils),
!--    this calculation is obsolete, as r_canopy is not used below.
!--    To do: check for very dry soil -> r_canopy goes to infinity.
       surf%r_canopy(m) = surf%r_canopy_min(m) /                                                   &
                          ( surf%lai(m) * f1(m) * f2(m) * f3(m) + 1.0E-20_wp )

!
!--    Calculate bare soil resistance r_soil.
       m_min = surf%c_veg(m) * surf%m_wilt(nzb_soil,m) +                                           &
               ( 1.0_wp - surf%c_veg(m) ) * surf%m_res(nzb_soil,m)
!
!--    Compute soil resistance.
       f2(m) = ( surf_m_soil%var_2d(nzb_soil,m) - m_min ) / ( surf%m_fc(nzb_soil,m) - m_min )
       f2(m) = MAX( f2(m), 1.0E-20_wp )
       f2(m) = MIN( f2(m), 1.0_wp     )

       surf%r_soil(m) = surf%r_soil_min(m) / f2(m)
    ENDDO

!
!-- Calculate net radiation radiation without longwave outgoing flux because it has a dependency
!-- on surface temperature and thus enters the prognostic equations directly.
    surf%rad_net_l = surf%rad_sw_in - surf%rad_sw_out + surf%rad_lw_in


    !$OMP PARALLEL DO PRIVATE (m, i, j, k, lambda_h_sat, ke, lambda_soil, lambda_surface,          &
    !$OMP&                     c_surface_tmp, m_liq_max, q_s, f_qsws_veg, f_qsws_soil, f_qsws_liq, &
    !$OMP&                     f_shf, f_qsws, e_s_dt, dq_s_dt, coef_1, coef_2, tend,               &
    !$OMP&                     i_off, j_off, k_off ) SCHEDULE (STATIC)
    DO  m = 1, surf%ns
!
!--    Index offset of surface element point with respect to adjoining atmospheric grid point
       k_off = surf%koff(m)
       j_off = surf%joff(m)
       i_off = surf%ioff(m)

       i = surf%i(m)
       j = surf%j(m)
       k = surf%k(m)

       horizontal = ( surf%upward(m)  .OR.  surf%downward(m) )
!
!--    Define heat conductivity between surface and soil depending on surface type. For vegetation,
!--    a skin layer parameterization is used. The new parameterization uses a combination of two
!--    conductivities: a constant conductivity for the skin layer, and a conductivity according to
!--    the uppermost soil layer. For bare soil and pavements, no skin layer is applied. In these
!--    cases, the temperature is assumed to be constant between the surface and the first soil
!--    layer. The heat conductivity is then derived from the soil/pavement properties.
!--    For water surfaces, the conductivity is already set to 1E10.
!--    Moreover, the heat capacity is set. For bare soil the heat capacity is the capacity of the
!--    uppermost soil layer, for pavement it is that of the material involved.
!--    For vegetation type surfaces, the thermal conductivity of the soil is needed.
       IF ( surf%vegetation_surface(m) )  THEN

          lambda_h_sat = lambda_h_sm**( 1.0_wp - surf%m_sat(nzb_soil,m) ) *                        &
                         lambda_h_water**surf_m_soil%var_2d(nzb_soil,m)

          ke = 1.0_wp + LOG10( MAX( 0.1_wp, surf_m_soil%var_2d(nzb_soil,m) /                       &
                                            surf%m_sat(nzb_soil,m) ) )

          lambda_soil = ( ke * ( lambda_h_sat - lambda_h_dry ) + lambda_h_dry ) *                  &
                        ddz_soil(nzb_soil) * 2.0_wp

!
!--       When bare soil is set without a thermal conductivity (no skin layer), the heat capacity is
!--       that of the soil layer, otherwise it is a combination of the conductivities from the skin
!--       and the soil layer.
          IF ( surf%lambda_surface_s(m) == 0.0_wp )  THEN
            surf%c_surface(m) = ( rho_c_soil * ( 1.0_wp - surf%m_sat(nzb_soil,m) ) +               &
                                  rho_c_water * surf_m_soil%var_2d(nzb_soil,m) ) *                 &
                                  dz_soil(nzb_soil) * 0.5_wp
            lambda_surface = lambda_soil

          ELSEIF ( surf_t_surface%var_1d(m) >= surf_t_soil%var_2d(nzb_soil,m) )                    &
          THEN
             lambda_surface = surf%lambda_surface_s(m) * lambda_soil /                             &
                              ( surf%lambda_surface_s(m) + lambda_soil )
          ELSE
             lambda_surface = surf%lambda_surface_u(m) * lambda_soil /                             &
                              ( surf%lambda_surface_u(m) + lambda_soil )
          ENDIF
       ELSE
          lambda_surface = surf%lambda_surface_s(m)
       ENDIF

!
!--    Set heat capacity of the skin/surface. It is ususally zero when a skin layer is used, and
!--    non-zero otherwise.
       c_surface_tmp = surf%c_surface(m)
!
!--    Calculate the maximum possible liquid water amount on plants and bare surface. For vegetated
!--    surfaces, a maximum depth of 0.2 mm is assumed, while paved surfaces might hold up 1 mm of
!--    water. The liquid water fraction for paved surfaces is calculated after Masson (2000)
!--    (TEB model) and originates from Noilhan & Planton (1989), while the ECMWF formulation is
!--    used for vegetated surfaces and bare soils.
       IF ( surf%pavement_surface(m) )  THEN
          m_liq_max = m_max_depth * 5.0_wp
          surf%c_liq(m) = MIN( 1.0_wp, ( surf_m_liq%var_1d(m) / m_liq_max )**0.67 )
       ELSE
          m_liq_max = m_max_depth * ( surf%c_veg(m) * surf%lai(m) +                                &
                      ( 1.0_wp - surf%c_veg(m) ) )
          surf%c_liq(m) = MIN( 1.0_wp, surf_m_liq%var_1d(m) / m_liq_max )
       ENDIF
!
!--    Calculate saturation water vapor mixing ratio.
       q_s = rd_d_rv * e_s(m) / ( surface_pressure - e_s(m) )
!
!--    In case of dewfall, set evapotranspiration to zero.
!--    All super-saturated water is then removed from the air.
       IF ( humidity  .AND.  q_s <= surf%qv1(m) )  THEN
          surf%r_canopy(m) = 0.0_wp
          surf%r_soil(m)   = 0.0_wp
       ENDIF
!
!--    Calculate coefficients for the total evapotranspiration.
!--    In case of water surface, set vegetation and soil fluxes to zero.
!--    For pavements, only evaporation of liquid water is possible.
       IF ( surf%water_surface(m) )  THEN
          f_qsws_veg  = 0.0_wp
          f_qsws_soil = 0.0_wp
          f_qsws_liq  = rho_lv / surf%r_a(m)
       ELSEIF ( surf%pavement_surface(m) )  THEN
          f_qsws_veg  = 0.0_wp
          f_qsws_soil = 0.0_wp
          f_qsws_liq  = rho_lv * surf%c_liq(m) / surf%r_a(m)
       ELSE
          f_qsws_veg  = rho_lv * surf%c_veg(m) *                                                   &
                            ( 1.0_wp        - surf%c_liq(m)  ) /                                   &
                            ( surf%r_a(m) + surf%r_canopy(m) )
          f_qsws_soil = rho_lv * (1.0_wp    - surf%c_veg(m)  )                                     &
                               * (1.0_wp    - surf%c_liq(m)  ) /                                   &
                            ( surf%r_a(m) + surf%r_soil(m)   )
          f_qsws_liq  = rho_lv * surf%c_liq(m) / surf%r_a(m)
       ENDIF

       f_shf  = rho_cp / surf%r_a(m)
       f_qsws = f_qsws_veg + f_qsws_soil + f_qsws_liq
!
!--    Calculate derivative of q_s for Taylor series expansion
       e_s_dt = e_s(m) * ( 17.62_wp / ( surf_t_surface%var_1d(m) -  29.65_wp ) -                   &
                           17.62_wp * ( surf_t_surface%var_1d(m) - 273.15_wp ) /                   &
                           ( surf_t_surface%var_1d(m) - 29.65_wp)**2 )

       dq_s_dt = rd_d_rv * e_s_dt / ( surface_pressure - e_s_dt )

!
!--    Calculate new skin temperature
       IF ( humidity )  THEN
!
!--       Numerator of the prognostic equation
          coef_1 = surf%rad_net_l(m) + surf%rad_lw_out_change_0(m) * surf_t_surface%var_1d(m)      &
                   - surf%rad_lw_out(m) + f_shf * surf%pt1(m)                                      &
                   + f_qsws * ( surf%qv1(m) - q_s + dq_s_dt * surf_t_surface%var_1d(m) )           &
                   + lambda_surface * surf_t_soil%var_2d(nzb_soil,m)

!
!--       Denominator of the prognostic equation
          coef_2 = surf%rad_lw_out_change_0(m) + f_qsws * dq_s_dt + lambda_surface                 &
                   + f_shf / exner(nzb)
       ELSE
!
!--       Numerator of the prognostic equation
          coef_1 = surf%rad_net_l(m) + surf%rad_lw_out_change_0(m) * surf_t_surface%var_1d(m)      &
                   - surf%rad_lw_out(m) + f_shf * surf%pt1(m)                                      &
                   + lambda_surface * surf_t_soil%var_2d(nzb_soil,m)
!
!--       Denominator of the prognostic equation
          coef_2 = surf%rad_lw_out_change_0(m) + lambda_surface + f_shf / exner(nzb)

       ENDIF

       tend = 0.0_wp
!
!--    Implicit solution when the surface layer has no heat capacity, otherwise use RK3 scheme.
       surf_t_surface_p%var_1d(m) = ( coef_1 * dt_3d * tsc(2) + c_surface_tmp *                    &
                                      surf_t_surface%var_1d(m) ) / ( c_surface_tmp + coef_2 *      &
                                                                     dt_3d * tsc(2) )
!
!--    Add RK3 term
       IF ( c_surface_tmp /= 0.0_wp )  THEN

          surf_t_surface_p%var_1d(m) = surf_t_surface_p%var_1d(m) + dt_3d * tsc(3) *               &
                                                                    surf_tt_surface_m%var_1d(m)

!
!--       Calculate true tendency
          tend = ( surf_t_surface_p%var_1d(m) - surf_t_surface%var_1d(m) -                         &
                   dt_3d * tsc(3) * surf_tt_surface_m%var_1d(m) ) / ( dt_3d  * tsc(2) )
!
!--       Calculate t_surface tendencies for the next Runge-Kutta step
          IF ( runge_l )  THEN
             IF ( intermediate_timestep_count == 1 )  THEN
                surf_tt_surface_m%var_1d(m) = tend
             ELSEIF ( intermediate_timestep_count < intermediate_timestep_count_max )  THEN
                surf_tt_surface_m%var_1d(m) = -9.5625_wp * tend +                                  &
                                               5.3125_wp * surf_tt_surface_m%var_1d(m)
             ENDIF
          ENDIF
       ENDIF

!
!--    In case of fast changes in the skin temperature, it is possible to update the radiative
!--    fluxes independently from the prescribed radiation call frequency. This effectively prevents
!--    oscillations, especially when setting skip_time_do_radiation /= 0. The threshold value of 0.2
!--    used here is just a first guess. This method should be revised in the future as tests have
!--    shown that the threshold is often reached, when no oscillations would occur (causes immense
!--    computing time for the radiation code).
       IF ( ABS( surf_t_surface_p%var_1d(m) - surf_t_surface%var_1d(m) ) > 0.2_wp  .AND.           &
            unscheduled_radiation_calls )  THEN
          force_radiation_call_l_v(m) = .TRUE.
       ENDIF

       surf%pt_surface(m) = surf_t_surface_p%var_1d(m) / exner(nzb)
!
!--    Calculate fluxes
       surf%rad_net_l(m) = surf%rad_net_l(m)                                                       &
                           + surf%rad_lw_out_change_0(m) * surf_t_surface%var_1d(m)                &
                           - surf%rad_lw_out(m)                                                    &
                           - surf%rad_lw_out_change_0(m) * surf_t_surface_p%var_1d(m)

       surf%rad_net(m) = surf%rad_net_l(m)
       surf%rad_lw_out(m) = surf%rad_lw_out(m) + surf%rad_lw_out_change_0(m) *                     &
                            ( surf_t_surface_p%var_1d(m) - surf_t_surface%var_1d(m) )

       surf%ghf(m) = lambda_surface * ( surf_t_surface_p%var_1d(m)                                 &
                                        - surf_t_soil%var_2d(nzb_soil,m) )

       surf%shf(m) = - f_shf * ( surf%pt1(m) - surf%pt_surface(m) ) / c_p
!
!--    Following line is necessary to remove the density from the flux. For horizontal surfaces
!--    where the heat flux is added to the vertical diffusion term, density cancels out in
!--    diffusion_s.f90. However, for vertical surfaces the density would still be included in the
!--    diffusion terms, meaning that the heat-fluxes at the walls would be overestimated by about
!--    15-20%.
       IF ( .NOT. horizontal )  surf%shf(m) = surf%shf(m) / rho_surface
!
!--    If applicable, Update fluxes based on DCEP
!--    ToDo: save shf and ghf befor editing for output
       IF ( dcep ) THEN
          surf%shf(m) = surf%shf(m) * ( 1.0_wp - fr_urb(j,i) )
          surf%ghf(m) = surf%ghf(m) * ( 1.0_wp - fr_urb(j,i) )
       ENDIF
!--    Add waste heat from anthropogenic heat sources to to the sensible heat flux 
       IF ( external_anthropogenic_heat  .AND.  ALLOCATED( surf%waste_heat ) )  THEN
         surf%shf(m) = surf%shf(m) + surf%waste_heat(m) / c_p  !! TODO: Check if the division by c_p is correct
       ENDIF
!
!--    Update the 3d field of rad_lw_out array to have consistent output
       IF ( surf%upward(m) )  THEN
          IF ( rrtmg_l )  THEN
             rad_lw_out(k+k_off,j+j_off,i+i_off) = surf%rad_lw_out(m)
          ELSE
             rad_lw_out(0,j+j_off,i+i_off) = surf%rad_lw_out(m)
          ENDIF
       ENDIF

       IF ( humidity )  THEN
          surf%qsws(m) = - f_qsws * ( surf%qv1(m) - q_s + dq_s_dt * surf_t_surface%var_1d(m)       &
                                      - dq_s_dt * surf_t_surface_p%var_1d(m) )

          surf%qsws_veg(m) = - f_qsws_veg  * ( surf%qv1(m) - q_s                                   &
                             + dq_s_dt * surf_t_surface%var_1d(m) - dq_s_dt                        &
                             * surf_t_surface_p%var_1d(m) )

          surf%qsws_soil(m) = - f_qsws_soil * ( surf%qv1(m) - q_s                                  &
                              + dq_s_dt * surf_t_surface%var_1d(m) - dq_s_dt                       &
                              * surf_t_surface_p%var_1d(m) )

          surf%qsws_liq(m) = - f_qsws_liq  * ( surf%qv1(m) - q_s                                   &
                             + dq_s_dt * surf_t_surface%var_1d(m) - dq_s_dt                        &
                             * surf_t_surface_p%var_1d(m) )
       ENDIF

!
!--    Calculate the true surface resistance. ABS is used here to avoid negative values that can
!--    occur for very small fluxes due to the artifical addition of 1.0E-20.
       IF ( .NOT.  humidity )  THEN
          surf%r_s(m) = 1.0E10_wp
       ELSE
          surf%r_s(m) = ABS( rho_lv / ( f_qsws + 1.0E-20_wp ) - surf%r_a(m) )
       ENDIF
!
!--    Calculate change in liquid water reservoir due to dew fall or evaporation of liquid water.
       IF ( humidity )  THEN
!
!--       If precipitation is activated, add rain water to qsws_liq and qsws_soil according to the
!--       vegetation coverage.
!--       precipitation_rate is given in mm.
          IF ( precipitation )  THEN

!
!--          Add precipitation to liquid water reservoir, if possible.
!--          Otherwise, add the water to soil. In case of pavements, the exceeding water amount is
!--          explicitly removed (as fictive runoff by drainage systems).
             IF ( surf%pavement_surface(m) )  THEN
                IF ( surf_m_liq%var_1d(m) < m_liq_max )  THEN
                   surf%qsws_liq(m) = surf%qsws_liq(m)                                             &
                                      + prr(k+k_off,j+j_off,i+i_off)                               &
                                      * hyrho(k+k_off)                                             &
                                      * 0.001_wp * rho_l * l_v
                ENDIF
             ELSE
                IF ( surf_m_liq%var_1d(m) < m_liq_max )  THEN
                   surf%qsws_liq(m) = surf%qsws_liq(m)                                             &
                                      + surf%c_veg(m) * prr(k+k_off,j+j_off,i+i_off)               &
                                      * hyrho(k+k_off)                                             &
                                      * 0.001_wp * rho_l * l_v
                   surf%qsws_soil(m) = surf%qsws_soil(m)                                           &
                                       + ( 1.0_wp - surf%c_veg(m) ) * prr(k+k_off,j+j_off,i+i_off) &
                                       * hyrho(k+k_off)                                            &
                                       * 0.001_wp * rho_l * l_v
                ELSE

!--                Add precipitation to bare soil according to the bare soil coverage.
                   surf%qsws_soil(m) = surf%qsws_soil(m)                                           &
                                       + surf%c_veg(m) * prr(k+k_off,j+j_off,i+i_off)              &
                                       * hyrho(k+k_off)                                            &
                                       * 0.001_wp * rho_l * l_v

                ENDIF
             ENDIF

          ENDIF

!
!--       If the air is saturated, check the reservoir water level.
          IF ( surf%qsws(m) < 0.0_wp )  THEN
!
!--          Check if reservoir is full (avoid values > m_liq_max).
!--          In that case, qsws_liq goes to qsws_soil for pervious surfaces. In this case qsws_veg
!--          is zero anyway (because c_liq = 1), so that tend is zero and no further check is
!--          needed.
             IF ( surf_m_liq%var_1d(m) == m_liq_max )  THEN
                IF ( .NOT. surf%pavement_surface(m))  THEN
                   surf%qsws_soil(m) = surf%qsws_soil(m) + surf%qsws_liq(m)
                ENDIF
                surf%qsws_liq(m)  = 0.0_wp
             ENDIF

!
!--          In case qsws_veg becomes negative (unphysical behavior), let the water enter the liquid
!--          water reservoir as dew on the plant.
             IF ( surf%qsws_veg(m) < 0.0_wp )  THEN
                surf%qsws_liq(m) = surf%qsws_liq(m) + surf%qsws_veg(m)
                surf%qsws_veg(m) = 0.0_wp
             ENDIF
          ENDIF

          surf%qsws(m) = surf%qsws(m) / l_v
!
!--       For explanation please see calculation of shf.
          IF ( .NOT. horizontal )  surf%qsws(m) = surf%qsws(m) / rho_surface

          tend = - surf%qsws_liq(m) * drho_l_lv
          surf_m_liq_p%var_1d(m) = surf_m_liq%var_1d(m) + dt_3d *                                  &
                                      ( tsc(2) * tend + tsc(3) * surf_tm_liq_m%var_1d(m) )
!
!--       Check if reservoir is overfull -> reduce to maximum (conservation of water is violated
!--       here).
          surf_m_liq_p%var_1d(m) = MIN( surf_m_liq_p%var_1d(m),m_liq_max )

!
!--       Check if reservoir is empty (avoid values < 0.0) (conservation of water is violated here).
          surf_m_liq_p%var_1d(m) = MAX( surf_m_liq_p%var_1d(m), 0.0_wp )
!
!--       Calculate m_liq tendencies for the next Runge-Kutta step
          IF ( runge_l )  THEN
             IF ( intermediate_timestep_count == 1 )  THEN
                surf_tm_liq_m%var_1d(m) = tend
             ELSEIF ( intermediate_timestep_count < intermediate_timestep_count_max )  THEN
                surf_tm_liq_m%var_1d(m) = -9.5625_wp * tend + 5.3125_wp * surf_tm_liq_m%var_1d(m)
             ENDIF
          ENDIF

       ENDIF

    ENDDO
    force_radiation_call_l = ANY(force_radiation_call_l_v)

!
!-- Make a logical OR for all processes. Force radiation call if at least one processor reached the
!-- threshold change in skin temperature.
    IF ( unscheduled_radiation_calls  .AND.                                                        &
         intermediate_timestep_count == intermediate_timestep_count_max-1 )  THEN
#if defined( __parallel )
       IF ( .NOT. force_radiation_call )  THEN
          IF ( collective_wait )  CALL MPI_BARRIER( comm2d, ierr )
          CALL MPI_ALLREDUCE( force_radiation_call_l, force_radiation_call, 1, MPI_LOGICAL,        &
                              MPI_LOR, comm2d, ierr )
       ENDIF
#else
       force_radiation_call = ( force_radiation_call  .OR.  force_radiation_call_l )
#endif
       force_radiation_call_l = .FALSE.
    ENDIF

!
!-- Calculate surface water vapor mixing ratio
    IF ( humidity )  THEN
       CALL calc_q_surface
    ENDIF
!
!-- Calculate new roughness lengths (for water surfaces only)
    IF ( .NOT. constant_roughness )  CALL calc_z0_water_surface

    IF ( debug_output_timestep )  THEN
       WRITE( debug_string, * ) 'lsm_surface_energy_balance'
       CALL debug_message( debug_string, 'end' )
    ENDIF

    CONTAINS
!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Calculation of mixing ratio of the skin layer (surface). It is assumend that the skin is always
!> saturated.
!--------------------------------------------------------------------------------------------------!
    SUBROUTINE calc_q_surface

       IMPLICIT NONE

       REAL(wp) ::  e_s           !< saturation water vapor pressure
       REAL(wp) ::  q_s           !< saturation mixing ratio
       REAL(wp) ::  resistance    !< aerodynamic and soil resistance term


       !$OMP PARALLEL DO PRIVATE (m, i, j, k, i_off, j_off, k_off, e_s, q_s, resistance) SCHEDULE (STATIC)
       DO  m = 1, surf%ns

          i = surf%i(m)
          j = surf%j(m)
          k = surf%k(m)
          i_off = surf%ioff(m)
          j_off = surf%joff(m)
          k_off = surf%koff(m)
!
!--       Calculate water vapour pressure at saturation and convert to hPa
          e_s = 0.01_wp * magnus( MIN(surf_t_surface_p%var_1d(m), 333.15_wp) )

!
!--       Calculate mixing ratio at saturation
          q_s = rd_d_rv * e_s / ( surface_pressure - e_s )

          resistance = surf%r_a(m) / ( surf%r_a(m) + surf%r_s(m) + 1E-5_wp )

!
!--       Calculate mixing ratio at surface
          IF ( bulk_cloud_model )  THEN
             q(k+k_off,j+j_off,i+i_off) = resistance * q_s +                                       &
                                          ( 1.0_wp - resistance ) * ( q(k,j,i) - ql(k,j,i) )
          ELSE
             q(k+k_off,j+j_off,i+i_off) = resistance * q_s +                                       &
                                          ( 1.0_wp - resistance ) * q(k,j,i)
          ENDIF

          surf%q_surface(m) = q(k+k_off,j+j_off,i+i_off)
!
!--       Update virtual potential temperature
          surf%vpt_surface(m) = surf%pt_surface(m) * ( 1.0_wp + 0.61_wp * surf%q_surface(m) )

       ENDDO

    END SUBROUTINE calc_q_surface

 END SUBROUTINE lsm_surface_energy_balance



!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Header output for land surface model
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_header ( io )


    IMPLICIT NONE

    CHARACTER (LEN=20) ::  coor_chr            !< Temporary string
    CHARACTER (LEN=86) ::  m_soil_chr          !< String for soil moisture
    CHARACTER (LEN=86) ::  roots_chr           !< String for root profile
    CHARACTER (LEN=86) ::  soil_depth_chr      !< String for soil depth
    CHARACTER (LEN=86) ::  t_soil_chr          !< String for soil temperature profile
    CHARACTER (LEN=86) ::  vertical_index_chr  !< String for the vertical index

    INTEGER(iwp) ::  i                         !< Loop index over soil layers

    INTEGER(iwp), INTENT(IN) ::  io            !< Unit of the output file


    t_soil_chr = ''
    m_soil_chr    = ''
    soil_depth_chr  = ''
    roots_chr        = ''
    vertical_index_chr   = ''

    i = 1
    DO i = nzb_soil, nzt_soil

       WRITE (coor_chr,'(F10.2,7X)') soil_temperature(i)
       t_soil_chr = TRIM( t_soil_chr ) // ' ' // TRIM( coor_chr )

       WRITE (coor_chr,'(F10.2,7X)') soil_moisture(i)
       m_soil_chr = TRIM( m_soil_chr ) // ' ' // TRIM( coor_chr )

       WRITE (coor_chr,'(F10.2,7X)')  - zs(i)
       soil_depth_chr = TRIM( soil_depth_chr ) // ' '  // TRIM( coor_chr )

       WRITE (coor_chr,'(F10.2,7X)')  root_fraction(i)
       roots_chr = TRIM( roots_chr ) // ' ' // TRIM( coor_chr )

       WRITE (coor_chr,'(I10,7X)')  i
       vertical_index_chr = TRIM( vertical_index_chr ) // ' ' // TRIM( coor_chr )

    ENDDO

!
!-- Write land surface model header
    WRITE( io,  1 )
    IF ( conserve_water_content )  THEN
       WRITE( io, 2 )
    ELSE
       WRITE( io, 3 )
    ENDIF

    IF ( vegetation_type_f%from_file )  THEN
       WRITE( io, 5 )
    ELSE
       WRITE( io, 4 ) TRIM( vegetation_type_name(vegetation_type) ),                               &
                      TRIM (soil_type_name(soil_type) )
    ENDIF
    WRITE( io, 6 ) TRIM( soil_depth_chr ), TRIM( t_soil_chr ), TRIM( m_soil_chr ),                 &
                      TRIM( roots_chr ), TRIM( vertical_index_chr )

1   FORMAT (//' Land surface model information:'/' ------------------------------'/)
2   FORMAT ('    --> Soil bottom is closed (water content is conserved',', default)')
3   FORMAT ('    --> Soil bottom is open (water content is not conserved)')
4   FORMAT ('    --> Land surface type  : ',A,/'    --> Soil porosity type : ',A)
5   FORMAT ('    --> Land surface type  : read from file'/                                         &
         '    --> Soil porosity type : read from file' )
6   FORMAT (/'    Initial soil temperature and moisture profile:'//                                &
         '       Height:        ',A,'  m'/                                                         &
         '       Temperature:   ',A,'  K'/                                                         &
         '       Moisture:      ',A,'  m**3/m**3'/                                                 &
         '       Root fraction: ',A,'  '/                                                          &
         '       Grid point:    ',A)


 END SUBROUTINE lsm_header


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Initialization of the land surface model
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_init

    USE control_parameters,                                                                        &
        ONLY:  message_string,                                                                     &
               nested_run

    USE indices,                                                                                   &
        ONLY:  nx,                                                                                 &
               ny,                                                                                 &
               topo_min_level

#if defined( __parallel )
    USE pmc_interface,                                                                             &
        ONLY:  root_model
#endif

    IMPLICIT NONE

    INTEGER(iwp) ::  exit_index              !< to store surface element index where z0 limit is exceeded
    INTEGER(iwp) ::  i                       !< running index x-direction
    INTEGER(iwp) ::  i_off                   !< offset index of surface element grid point and referece grid point (in x)
    INTEGER(iwp) ::  j                       !< running index y-direction
    INTEGER(iwp) ::  j_off                   !< offset index of surface element grid point and referece grid point (in y)
    INTEGER(iwp) ::  k                       !< running index z-direction
    INTEGER(iwp) ::  kn                      !< running index
    INTEGER(iwp) ::  ko                      !< running index
    INTEGER(iwp) ::  kroot                   !< running index
    INTEGER(iwp) ::  kzs                     !< running index
    INTEGER(iwp) ::  m                       !< running index surface elements
    INTEGER(iwp) ::  st                      !< soil-type index
    INTEGER(iwp) ::  n_soil_layers_total     !< temperature variable, stores the total number of soil layers + 4
#if defined( __parallel )
    INTEGER(iwp) ::  nzs_root                !< number of soil layers in root domain (used in case soil data needs to be
                                             !< transferred from root model to child models)

    LOGICAL ::  init_msoil_from_driver_root !< flag indicating that msoil in root is initialized from dynamic file
    LOGICAL ::  init_tsoil_from_driver_root !< flag indicating that tsoil in root is initialized from dynamic file
#endif
    LOGICAL ::  flag_negative_value         !< dummy flag to check for negative soil moisture
    LOGICAL ::  flag_exceed_sat             !< dummy flag to check for saturation moisture
    LOGICAL ::  flag_exceed_z0              !< dummy flag to indicate whether roughness length is too high
    LOGICAL ::  flag_exceed_z0h             !< dummy flag to indicate whether roughness length for temperature is too high
    LOGICAL ::  flag_exceed_z0q             !< dummy flag to indicate whether roughness length for moisture is too high

#if defined( __parallel )
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  m_soil_root    !< domain-averaged soil moisture profile in root domain
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  t_soil_root    !< domain-averaged soil temperature profile in root domain
#endif
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  bound          !< temporary arrays for storing index bounds
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  bound_root_fr  !< temporary arrays for storing index bounds
#if defined( __parallel )
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  pr_soil_init   !< temporary array used for averaging soil profiles
    REAL(wp), DIMENSION(:), ALLOCATABLE ::  z_soil_root    !< vertical dimension of soil grid in root domain
#endif

    IF ( debug_output )  CALL debug_message( 'lsm_init', 'start' )
!
!-- If no cloud physics is used, rho_surface has not been calculated before.
    IF ( .NOT. bulk_cloud_model  .AND.  .NOT. cloud_droplets )  THEN
       CALL calc_mean_profile( pt, 4 )
       rho_surface = hyp(nzb) / ( r_d * hom(topo_min_level+1,1,4,0) * exner(nzb) )
    ENDIF

!
!-- Store soil-layer depth and indices on surface structure.
    ALLOCATE( surf_lsm%zs(nzb_soil:nzt_soil) )
    surf_lsm%zs(nzb_soil:nzt_soil) = zs(nzb_soil:nzt_soil)

    surf_lsm%nzb_soil = nzb_soil
    surf_lsm%nzt_soil = nzt_soil

!
!-- Calculate frequently used parameters
    rho_cp    = c_p * rho_surface
    rho_lv    = rho_surface * l_v
    drho_l_lv = 1.0_wp / (rho_l * l_v)

!
!-- Set initial values for prognostic quantities
!-- Horizontal surfaces
    tt_surface_m%var_1d = 0.0_wp
    tt_soil_m%var_2d    = 0.0_wp
    tm_soil_m%var_2d    = 0.0_wp
    tm_liq_m%var_1d     = 0.0_wp
    surf_lsm%c_liq      = 0.0_wp

    surf_lsm%ghf = 0.0_wp

    surf_lsm%qsws_liq  = 0.0_wp
    surf_lsm%qsws_soil = 0.0_wp
    surf_lsm%qsws_veg  = 0.0_wp

    surf_lsm%r_a        = 50.0_wp
    surf_lsm%r_s        = 50.0_wp
    surf_lsm%r_canopy   = 0.0_wp
    surf_lsm%r_soil     = 0.0_wp

!
!-- Set initial values for prognostic soil quantities. This must not be done in restart runs or
!-- when spinup data is read.
    IF ( TRIM( initializing_actions ) /= 'read_restart_data'  .AND.  .NOT. read_spinup_data )  THEN
       t_soil%var_2d = 0.0_wp
       m_soil%var_2d = 0.0_wp
       m_liq%var_1d  = 0.0_wp
    ENDIF
!
!-- Allocate 3D soil model arrays
    ALLOCATE ( surf_lsm%alpha_vg(nzb_soil:nzt_soil,1:surf_lsm%ns) )
    ALLOCATE ( surf_lsm%gamma_w_sat(nzb_soil:nzt_soil,1:surf_lsm%ns) )
    ALLOCATE ( surf_lsm%lambda_h(nzb_soil:nzt_soil,1:surf_lsm%ns) )
    ALLOCATE ( surf_lsm%lambda_h_def(nzb_soil:nzt_soil,1:surf_lsm%ns) )
    ALLOCATE ( surf_lsm%l_vg(nzb_soil:nzt_soil,1:surf_lsm%ns) )
    ALLOCATE ( surf_lsm%m_fc(nzb_soil:nzt_soil,1:surf_lsm%ns) )
    ALLOCATE ( surf_lsm%m_res(nzb_soil:nzt_soil,1:surf_lsm%ns) )
    ALLOCATE ( surf_lsm%m_sat(nzb_soil:nzt_soil,1:surf_lsm%ns) )
    ALLOCATE ( surf_lsm%m_wilt(nzb_soil:nzt_soil,1:surf_lsm%ns) )
    ALLOCATE ( surf_lsm%n_vg(nzb_soil:nzt_soil,1:surf_lsm%ns) )
    ALLOCATE ( surf_lsm%rho_c_total(nzb_soil:nzt_soil,1:surf_lsm%ns) )
    ALLOCATE ( surf_lsm%rho_c_total_def(nzb_soil:nzt_soil,1:surf_lsm%ns) )
    ALLOCATE ( surf_lsm%root_fr(nzb_soil:nzt_soil,1:surf_lsm%ns) )

    surf_lsm%lambda_h     = 0.0_wp
!
!-- If required, allocate humidity-related variables for the soil model
    IF ( humidity )  THEN
       ALLOCATE ( surf_lsm%lambda_w(nzb_soil:nzt_soil,1:surf_lsm%ns) )
       ALLOCATE ( surf_lsm%gamma_w(nzb_soil:nzt_soil,1:surf_lsm%ns) )

       surf_lsm%lambda_w = 0.0_wp
    ENDIF
!
!-- Allocate albedo type and emissivity for vegetation, water and pavement fraction.
!-- Set default values at each surface element.
    ALLOCATE ( surf_lsm%albedo_type(1:surf_lsm%ns,0:2) )
    ALLOCATE ( surf_lsm%emissivity(1:surf_lsm%ns,0:2) )
!
!-- Initialize albedo type according to its default type, in order to set values independently
!-- on default albedo_type in radiation model.
    surf_lsm%albedo_type(:,ind_veg_wall)  = INT( vegetation_pars(ind_v_at,vegetation_type) )
    surf_lsm%albedo_type(:,ind_wat_win)   = INT( water_pars(ind_w_at,water_type) )
    surf_lsm%albedo_type(:,ind_pav_green) = INT( pavement_pars(ind_p_at,pavement_type) )
    surf_lsm%emissivity  = emissivity
!
!-- Allocate arrays for relative surface fraction.
!-- 0 - vegetation fraction, 2 - water fraction, 1 - pavement fraction
    ALLOCATE( surf_lsm%frac(1:surf_lsm%ns,0:2) )
    surf_lsm%frac = 0.0_wp
!
!-- For vertical walls only - allocate special flag indicating if any building is on top of any
!-- natural surfaces. Used for initialization only.
    ALLOCATE( surf_lsm%building_covered(1:surf_lsm%ns) )
!
!-- Allocate arrays for the respective types and their names on the surface elements. This will be
!-- required to treat deposition of chemical species.
    ALLOCATE( surf_lsm%pavement_type(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%vegetation_type(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%water_type(1:surf_lsm%ns) )

    surf_lsm%pavement_type   = 0
    surf_lsm%vegetation_type = 0
    surf_lsm%water_type      = 0

    ALLOCATE( surf_lsm%pavement_type_name(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%vegetation_type_name(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%water_type_name(1:surf_lsm%ns) )

    surf_lsm%pavement_type_name   = 'none'
    surf_lsm%vegetation_type_name = 'none'
    surf_lsm%water_type_name      = 'none'
!
!-- Set flag parameter for the prescribed surface type depending on user input. Set surface fraction
!-- to 1 for the respective type.
    SELECT CASE ( TRIM( surface_type ) )

       CASE ( 'vegetation' )
          surf_lsm%vegetation_surface = .TRUE.
          surf_lsm%frac(:,ind_veg_wall) = 1.0_wp

       CASE ( 'water' )
!
!--       Note, downward and vertical water surface does not really make sense.
          surf_lsm%water_surface = .TRUE.
          surf_lsm%frac(:,ind_wat_win) = 1.0_wp

       CASE ( 'pavement' )
          surf_lsm%pavement_surface = .TRUE.
          surf_lsm%frac(:,ind_pav_green) = 1.0_wp

       CASE ( 'netcdf', 'read_from_file' )
          DO  m = 1, surf_lsm%ns
             i = surf_lsm%i(m)
             j = surf_lsm%j(m)
             i_off = surf_lsm%ioff(m)
             j_off = surf_lsm%joff(m)
!
!--          For vertical surface set a special flag which determines the reference grid point
!--          from which input data should be taken. Therefore, check if at the grid point where
!--          the surface is defined (i+ioff, j+joff) is any building.
!--          This case, natural surfaces properties are not necessarily defined at this grid point,
!--          leading to problems in the initialization.
!--          Further, there can occur a special case where elevation changes are larger than
!--          building heights. This case, (j,i) and (j+joff,i+ioff) grid points may be both
!--          covered by buildings, but vertical natural walls may be located between the
!--          buildings. This case, it is not guaranteed that information about natural surface
!--          types is given, neither at (j,i) nor at (j+joff,i+ioff), again leading to
!--          non-initialized surface properties.
             surf_lsm%building_covered(m) = .FALSE.
!
!--          Wall grid point is building-covered. This case, set flag indicating that surface
!--          properties are initialized from neighboring reference grid point, which is not
!--          building_covered.
             IF ( building_type_f%from_file )  THEN
                IF ( building_type_f%var(j+j_off,i+i_off) /= building_type_f%fill )                &
                   surf_lsm%building_covered(m) = .TRUE.
!
!--             Wall grid point as well as neighboring reference grid point are both
!--             building-covered. This case, surface properties are not necessarily defined (not
!--             covered by checks for static input file) at this surface. Hence, initialize
!--             surface properties by simply setting vegetation_type_f to bare-soil bulk
!--             parametrization.
!--             soil_type_f as well as surface_fractions_f will be set also.
                IF ( building_type_f%var(j+j_off,i+i_off)  /= building_type_f%fill  .AND.          &
                     building_type_f%var(j,i)              /= building_type_f%fill )               &
                THEN
                   vegetation_type_f%var(j,i) = 1 ! bare soil
                   soil_type_f%var_2d(j,i)    = 1

                   water_type_f%var(j,i)      = water_type_f%fill
                   pavement_type_f%var(j,i)   = pavement_type_f%fill
!
!--                If surface_fraction is provided in static input, set fraction for vegetation
!--                to one at building-covered surfaces.
                   IF ( surface_fraction_f%from_file )  THEN
                      surface_fraction_f%frac(ind_veg_wall,j,i)  = 1.0_wp
                      surface_fraction_f%frac(ind_pav_green,j,i) = 0.0_wp
                      surface_fraction_f%frac(ind_wat_win,j,i)   = 0.0_wp
                   ENDIF
                ENDIF

             ENDIF
!
!--          Set reference grid point from which the surface gets classified. For building-covered
!--          grid points take the classification at the given
             i = surf_lsm%i(m) + MERGE( 0, i_off, surf_lsm%building_covered(m) )
             j = surf_lsm%j(m) + MERGE( 0, j_off, surf_lsm%building_covered(m) )

             IF ( vegetation_type_f%var(j,i) /= vegetation_type_f%fill )                           &
                surf_lsm%vegetation_surface(m) = .TRUE.
             IF ( pavement_type_f%var(j,i)   /= pavement_type_f%fill )                             &
                surf_lsm%pavement_surface(m) = .TRUE.
             IF ( water_type_f%var(j,i)      /= water_type_f%fill )                                &
                surf_lsm%water_surface(m) = .TRUE.
!
!--          Check if at least one type is set.
             IF ( .NOT. surf_lsm%vegetation_surface(m)  .AND.                                      &
                  .NOT. surf_lsm%pavement_surface(m)    .AND.                                      &
                  .NOT. surf_lsm%water_surface(m) )  THEN
                WRITE( message_string, * ) 'surface element at grid point (i,j) = (', i, j, ') ',  &
                                           'is neither a vegetation, pavement, nor a water surface'
                CALL message( 'land_surface_model_mod', 'LSM0039', 2, 2, myid, 6, 0 )
             ENDIF
          ENDDO

    END SELECT
!
!-- In case of static file input, further initialize surface fractions.
!-- At the moment only 1 surface is given at a location, so that the fraction is either 0 or 1.
!-- This will be revised later. If surface fraction is not given in static input file, relative
!-- fractions will be derived from given surface type. In this case, only 1 type is given at a
!-- certain location (already checked).
    IF ( input_pids_static  .AND.  surface_fraction_f%from_file )  THEN
       DO  m = 1, surf_lsm%ns
          i = surf_lsm%i(m) + MERGE( 0, surf_lsm%ioff(m), surf_lsm%building_covered(m) )
          j = surf_lsm%j(m) + MERGE( 0, surf_lsm%joff(m), surf_lsm%building_covered(m) )
!
!--       0 - vegetation fraction, 1 - pavement fraction, 2 - water fraction
          IF ( surface_fraction_f%frac(ind_veg_wall,j,i) /= surface_fraction_f%fill )  THEN
             surf_lsm%frac(m,ind_veg_wall)  = surface_fraction_f%frac(ind_veg_wall,j,i)
          ENDIF
          IF ( surface_fraction_f%frac(ind_pav_green,j,i) /= surface_fraction_f%fill )  THEN
             surf_lsm%frac(m,ind_pav_green) = surface_fraction_f%frac(ind_pav_green,j,i)
          ENDIF
          IF ( surface_fraction_f%frac(ind_wat_win,j,i) /= surface_fraction_f%fill )  THEN
             surf_lsm%frac(m,ind_wat_win)   = surface_fraction_f%frac(ind_wat_win,j,i)
          ENDIF
!
!--       Check if sum of relative fractions is zero. This case, give an error message.
          IF ( SUM ( surf_lsm%frac(m,:) ) == 0.0_wp )  THEN
             WRITE( message_string, * ) 'surface fractions at grid point (i,j) = (',               &
                                        i, j, ') are all zero'
             CALL message( 'land_surface_model_mod', 'LSM0040', 2, 2, myid, 6, 0 )
          ENDIF
!
!--       In case the sum of all surfaces is not 1, which may happen due to rounding errors or
!--       type conversions, normalize the fractions to one. Note, at the moment no tile approach
!--       is implemented, so that relative fractions are either 1 or zero.
          IF ( SUM ( surf_lsm%frac(m,:) ) > 1.0_wp  .OR.  SUM ( surf_lsm%frac(m,:) ) < 1.0_wp  )   &
             surf_lsm%frac(m,:) = surf_lsm%frac(m,:) / SUM ( surf_lsm%frac(m,:) )
       ENDDO

    ELSEIF ( input_pids_static  .AND.  .NOT. surface_fraction_f%from_file )  THEN
       DO  m = 1, surf_lsm%ns
          i = surf_lsm%i(m) + MERGE( 0, surf_lsm%ioff(m), surf_lsm%building_covered(m) )
          j = surf_lsm%j(m) + MERGE( 0, surf_lsm%joff(m), surf_lsm%building_covered(m) )

          IF ( vegetation_type_f%var(j,i) /= vegetation_type_f%fill )                              &
             surf_lsm%frac(m,ind_veg_wall)  = 1.0_wp
          IF ( pavement_type_f%var(j,i)   /= pavement_type_f%fill   )                              &
             surf_lsm%frac(m,ind_pav_green) = 1.0_wp
          IF ( water_type_f%var(j,i)      /= water_type_f%fill      )                              &
             surf_lsm%frac(m,ind_wat_win)   = 1.0_wp
       ENDDO

    ENDIF
!
!-- Level 1, initialization of soil parameters.
!-- It is possible to overwrite each parameter by setting the respecticy NAMELIST variable to a
!-- value /= 9999999.9.
    IF ( soil_type /= 0 )  THEN

       IF ( alpha_vangenuchten == 9999999.9_wp )     alpha_vangenuchten = soil_pars(0,soil_type)
       IF ( l_vangenuchten == 9999999.9_wp )         l_vangenuchten = soil_pars(1,soil_type)
       IF ( n_vangenuchten == 9999999.9_wp )         n_vangenuchten = soil_pars(2,soil_type)
       IF ( hydraulic_conductivity == 9999999.9_wp ) hydraulic_conductivity = soil_pars(3,soil_type)
       IF ( saturation_moisture == 9999999.9_wp )    saturation_moisture = soil_pars(4,soil_type)
       IF ( field_capacity == 9999999.9_wp )         field_capacity = soil_pars(5,soil_type)
       IF ( wilting_point == 9999999.9_wp )          wilting_point = soil_pars(6,soil_type)
       IF ( residual_moisture == 9999999.9_wp )      residual_moisture = soil_pars(7,soil_type)

    ENDIF
!
!-- Map values to the respective 2D/3D arrays
    surf_lsm%alpha_vg    = alpha_vangenuchten
    surf_lsm%l_vg        = l_vangenuchten
    surf_lsm%n_vg        = n_vangenuchten
    surf_lsm%gamma_w_sat = hydraulic_conductivity
    surf_lsm%m_sat       = saturation_moisture
    surf_lsm%m_fc        = field_capacity
    surf_lsm%m_wilt      = wilting_point
    surf_lsm%m_res       = residual_moisture
    surf_lsm%r_soil_min  = min_soil_resistance
!
!-- Level 2, initialization of soil parameters via soil_type read from file.
!-- Soil parameters are initialized for each (y,x)-grid point individually using default parameter
!-- settings according to the given soil type.
    IF ( soil_type_f%from_file )  THEN
!
!--    Level of detail = 1, i.e. a homogeneous soil distribution along the vertical dimension is
!--    assumed.
       IF ( soil_type_f%lod == 1 )  THEN
          DO  m = 1, surf_lsm%ns
             i = surf_lsm%i(m) + MERGE( 0, surf_lsm%ioff(m), surf_lsm%building_covered(m) )
             j = surf_lsm%j(m) + MERGE( 0, surf_lsm%joff(m), surf_lsm%building_covered(m) )

             st = soil_type_f%var_2d(j,i)
             IF ( st /= soil_type_f%fill )  THEN
                surf_lsm%alpha_vg(:,m)    = soil_pars(0,st)
                surf_lsm%l_vg(:,m)        = soil_pars(1,st)
                surf_lsm%n_vg(:,m)        = soil_pars(2,st)
                surf_lsm%gamma_w_sat(:,m) = soil_pars(3,st)
                surf_lsm%m_sat(:,m)       = soil_pars(4,st)
                surf_lsm%m_fc(:,m)        = soil_pars(5,st)
                surf_lsm%m_wilt(:,m)      = soil_pars(6,st)
                surf_lsm%m_res(:,m)       = soil_pars(7,st)
             ENDIF
          ENDDO
!
!--    Level of detail = 2, i.e. soil type and thus the soil parameters can be heterogeneous along
!--    the vertical dimension.
       ELSE

          DO  m = 1, surf_lsm%ns
             i = surf_lsm%i(m) + MERGE( 0, surf_lsm%ioff(m), surf_lsm%building_covered(m) )
             j = surf_lsm%j(m) + MERGE( 0, surf_lsm%joff(m), surf_lsm%building_covered(m) )

             DO  k = nzb_soil, nzt_soil
                st = soil_type_f%var_3d(k,j,i)
                IF ( st /= soil_type_f%fill )  THEN
                   surf_lsm%alpha_vg(k,m)    = soil_pars(0,st)
                   surf_lsm%l_vg(k,m)        = soil_pars(1,st)
                   surf_lsm%n_vg(k,m)        = soil_pars(2,st)
                   surf_lsm%gamma_w_sat(k,m) = soil_pars(3,st)
                   surf_lsm%m_sat(k,m)       = soil_pars(4,st)
                   surf_lsm%m_fc(k,m)        = soil_pars(5,st)
                   surf_lsm%m_wilt(k,m)      = soil_pars(6,st)
                   surf_lsm%m_res(k,m)       = soil_pars(7,st)
                ENDIF
             ENDDO
          ENDDO

       ENDIF
    ENDIF
!
!-- Level 3, initialization of single soil parameters at single z,x,y position via soil_pars read
!-- from file.
    IF ( soil_pars_f%from_file )  THEN
!
!--    Level of detail = 1, i.e. a homogeneous vertical distribution of soil parameters is assumed.
!--    Horizontal surfaces
       IF ( soil_pars_f%lod == 1 )  THEN
          DO  m = 1, surf_lsm%ns
             i = surf_lsm%i(m) + MERGE( 0, surf_lsm%ioff(m), surf_lsm%building_covered(m) )
             j = surf_lsm%j(m) + MERGE( 0, surf_lsm%joff(m), surf_lsm%building_covered(m) )

             IF ( soil_pars_f%pars_xy(0,j,i) /= soil_pars_f%fill )                                 &
                surf_lsm%alpha_vg(:,m)    = soil_pars_f%pars_xy(0,j,i)
             IF ( soil_pars_f%pars_xy(1,j,i) /= soil_pars_f%fill )                                 &
                surf_lsm%l_vg(:,m)        = soil_pars_f%pars_xy(1,j,i)
             IF ( soil_pars_f%pars_xy(2,j,i) /= soil_pars_f%fill )                                 &
                surf_lsm%n_vg(:,m)        = soil_pars_f%pars_xy(2,j,i)
             IF ( soil_pars_f%pars_xy(3,j,i) /= soil_pars_f%fill )                                 &
                surf_lsm%gamma_w_sat(:,m) = soil_pars_f%pars_xy(3,j,i)
             IF ( soil_pars_f%pars_xy(4,j,i) /= soil_pars_f%fill )                                 &
                surf_lsm%m_sat(:,m)       = soil_pars_f%pars_xy(4,j,i)
             IF ( soil_pars_f%pars_xy(5,j,i) /= soil_pars_f%fill )                                 &
                surf_lsm%m_fc(:,m)        = soil_pars_f%pars_xy(5,j,i)
             IF ( soil_pars_f%pars_xy(6,j,i) /= soil_pars_f%fill )                                 &
                surf_lsm%m_wilt(:,m)      = soil_pars_f%pars_xy(6,j,i)
             IF ( soil_pars_f%pars_xy(7,j,i) /= soil_pars_f%fill )                                 &
                surf_lsm%m_res(:,m)       = soil_pars_f%pars_xy(7,j,i)
          ENDDO
!
!--    Level of detail = 2, i.e. soil parameters can be set at each soil layer individually.
       ELSE
          DO  m = 1, surf_lsm%ns
             i = surf_lsm%i(m) + MERGE( 0, surf_lsm%ioff(m), surf_lsm%building_covered(m) )
             j = surf_lsm%j(m) + MERGE( 0, surf_lsm%joff(m), surf_lsm%building_covered(m) )

             DO  k = nzb_soil, nzt_soil
                IF ( soil_pars_f%pars_xyz(0,k,j,i) /= soil_pars_f%fill )                           &
                   surf_lsm%alpha_vg(k,m)    = soil_pars_f%pars_xyz(0,k,j,i)
                IF ( soil_pars_f%pars_xyz(1,k,j,i) /= soil_pars_f%fill )                           &
                   surf_lsm%l_vg(k,m)        = soil_pars_f%pars_xyz(1,k,j,i)
                IF ( soil_pars_f%pars_xyz(2,k,j,i) /= soil_pars_f%fill )                           &
                   surf_lsm%n_vg(k,m)        = soil_pars_f%pars_xyz(2,k,j,i)
                IF ( soil_pars_f%pars_xyz(3,k,j,i) /= soil_pars_f%fill )                           &
                   surf_lsm%gamma_w_sat(k,m) = soil_pars_f%pars_xyz(3,k,j,i)
                IF ( soil_pars_f%pars_xyz(4,k,j,i) /= soil_pars_f%fill )                           &
                   surf_lsm%m_sat(k,m)       = soil_pars_f%pars_xyz(4,k,j,i)
                IF ( soil_pars_f%pars_xyz(5,k,j,i) /= soil_pars_f%fill )                           &
                   surf_lsm%m_fc(k,m)        = soil_pars_f%pars_xyz(5,k,j,i)
                IF ( soil_pars_f%pars_xyz(6,k,j,i) /= soil_pars_f%fill )                           &
                   surf_lsm%m_wilt(k,m)      = soil_pars_f%pars_xyz(6,k,j,i)
                IF ( soil_pars_f%pars_xyz(7,k,j,i) /= soil_pars_f%fill )                           &
                   surf_lsm%m_res(k,m)       = soil_pars_f%pars_xyz(7,k,j,i)
             ENDDO
          ENDDO
       ENDIF
    ENDIF

!
!-- Level 1, initialization of vegetation parameters. A horizontally homogeneous distribution is
!-- assumed here.
    IF ( vegetation_type /= 0 )  THEN

       IF ( min_canopy_resistance == 9999999.9_wp )  THEN
          min_canopy_resistance = vegetation_pars(ind_v_rc_min,vegetation_type)
       ENDIF

       IF ( leaf_area_index == 9999999.9_wp )  THEN
          leaf_area_index = vegetation_pars(ind_v_rc_lai,vegetation_type)
       ENDIF

       IF ( vegetation_coverage == 9999999.9_wp )  THEN
          vegetation_coverage = vegetation_pars(ind_v_c_veg,vegetation_type)
       ENDIF

       IF ( canopy_resistance_coefficient == 9999999.9_wp )  THEN
           canopy_resistance_coefficient= vegetation_pars(ind_v_gd,vegetation_type)
       ENDIF

       IF ( z0_vegetation == 9999999.9_wp )  THEN
          z0_vegetation  = vegetation_pars(ind_v_z0,vegetation_type)
       ENDIF

       IF ( z0h_vegetation == 9999999.9_wp )  THEN
          z0h_vegetation = vegetation_pars(ind_v_z0qh,vegetation_type)
       ENDIF

       IF ( z0q_vegetation == 9999999.9_wp )  THEN
          z0q_vegetation = vegetation_pars(ind_v_z0qh,vegetation_type)
       ENDIF

       IF ( lambda_surface_stable == 9999999.9_wp )  THEN
          lambda_surface_stable = vegetation_pars(ind_v_lambda_s,vegetation_type)
       ENDIF

       IF ( lambda_surface_unstable == 9999999.9_wp )  THEN
          lambda_surface_unstable = vegetation_pars(ind_v_lambda_u,vegetation_type)
       ENDIF

       IF ( f_shortwave_incoming == 9999999.9_wp )  THEN
          f_shortwave_incoming = vegetation_pars(ind_v_f_sw_in,vegetation_type)
       ENDIF

       IF ( c_surface == 9999999.9_wp )  THEN
          c_surface = vegetation_pars(ind_v_c_surf,vegetation_type)
       ENDIF

       IF ( albedo_type == 9999999  .AND.  albedo == 9999999.9_wp )  THEN
          albedo_type = INT(vegetation_pars(ind_v_at,vegetation_type))
       ENDIF

       IF ( emissivity == 9999999.9_wp )  THEN
          emissivity = vegetation_pars(ind_v_emis,vegetation_type)
       ENDIF

    ENDIF
!
!-- Map values onto surface elemements
    DO  m = 1, surf_lsm%ns
       IF ( surf_lsm%vegetation_surface(m) )  THEN
          surf_lsm%r_canopy_min(m)     = min_canopy_resistance
          surf_lsm%lai(m)              = leaf_area_index
          surf_lsm%c_veg(m)            = vegetation_coverage
          surf_lsm%g_d(m)              = canopy_resistance_coefficient
          surf_lsm%z0(m)               = z0_vegetation
          surf_lsm%z0h(m)              = z0h_vegetation
          surf_lsm%z0q(m)              = z0q_vegetation
          surf_lsm%lambda_surface_s(m) = lambda_surface_stable
          surf_lsm%lambda_surface_u(m) = lambda_surface_unstable
          surf_lsm%f_sw_in(m)          = f_shortwave_incoming
          surf_lsm%c_surface(m)        = c_surface
          surf_lsm%albedo_type(m,ind_veg_wall) = albedo_type
          surf_lsm%emissivity(m,ind_veg_wall)  = emissivity

          surf_lsm%vegetation_type(m)      = vegetation_type
          surf_lsm%vegetation_type_name(m) = vegetation_type_name(vegetation_type)
       ELSE
          surf_lsm%lai(m)   = 0.0_wp
          surf_lsm%c_veg(m) = 0.0_wp
          surf_lsm%g_d(m)   = 0.0_wp
       ENDIF
    ENDDO
!
!-- Level 2, initialization of vegation parameters via vegetation_type read from file. Vegetation
!-- parameters are initialized for each (y,x)-grid point individually using default paramter
!-- settings according to the given vegetation type.
    IF ( vegetation_type_f%from_file )  THEN
       DO  m = 1, surf_lsm%ns
          i = surf_lsm%i(m) + MERGE( 0, surf_lsm%ioff(m), surf_lsm%building_covered(m) )
          j = surf_lsm%j(m) + MERGE( 0, surf_lsm%joff(m), surf_lsm%building_covered(m) )

          st = vegetation_type_f%var(j,i)
          IF ( st /= vegetation_type_f%fill  .AND.  st /= 0 )  THEN
             surf_lsm%r_canopy_min(m)     = vegetation_pars(ind_v_rc_min,st)
             surf_lsm%lai(m)              = vegetation_pars(ind_v_rc_lai,st)
             surf_lsm%c_veg(m)            = vegetation_pars(ind_v_c_veg,st)
             surf_lsm%g_d(m)              = vegetation_pars(ind_v_gd,st)
             surf_lsm%z0(m)               = vegetation_pars(ind_v_z0,st)
             surf_lsm%z0h(m)              = vegetation_pars(ind_v_z0qh,st)
             surf_lsm%z0q(m)              = vegetation_pars(ind_v_z0qh,st)
             surf_lsm%lambda_surface_s(m) = vegetation_pars(ind_v_lambda_s,st)
             surf_lsm%lambda_surface_u(m) = vegetation_pars(ind_v_lambda_u,st)
             surf_lsm%f_sw_in(m)          = vegetation_pars(ind_v_f_sw_in,st)
             surf_lsm%c_surface(m)        = vegetation_pars(ind_v_c_surf,st)
             surf_lsm%albedo_type(m,ind_veg_wall) = INT( vegetation_pars(ind_v_at,st) )
             surf_lsm%emissivity(m,ind_veg_wall)  = vegetation_pars(ind_v_emis,st)

             surf_lsm%vegetation_type(m)      = st
             surf_lsm%vegetation_type_name(m) = vegetation_type_name(st)
          ENDIF
       ENDDO
    ENDIF
!
!-- Level 3, initialization of vegation parameters at single (x,y) position via vegetation_pars read
!-- from file.
    IF ( vegetation_pars_f%from_file )  THEN

       DO  m = 1, surf_lsm%ns
          i = surf_lsm%i(m) + MERGE( 0, surf_lsm%ioff(m), surf_lsm%building_covered(m) )
          j = surf_lsm%j(m) + MERGE( 0, surf_lsm%joff(m), surf_lsm%building_covered(m) )
!
!--       If surface element is not a vegetation surface and any value in vegetation_pars is
!--       given, neglect this information and give an informative message that this value will
!--       not be used.
          IF ( .NOT. surf_lsm%vegetation_surface(m)  .AND.                                         &
                ANY( vegetation_pars_f%pars_xy(:,j,i) /= vegetation_pars_f%fill ) )                &
          THEN
             WRITE( message_string, * ) 'surface element at grid point (i,j) = (',                 &
                                        i, j, ') is not a vegetation surface, ',                   &
                                        'so that information given in ',                           &
                                       'vegetation_pars at this point is neglected.'
             CALL message( 'land_surface_model_mod', 'LSM0041', 0, 0, myid, 6, 0 )
          ELSE

             IF ( vegetation_pars_f%pars_xy(ind_v_rc_min,j,i) /= vegetation_pars_f%fill )          &
                surf_lsm%r_canopy_min(m)  = vegetation_pars_f%pars_xy(ind_v_rc_min,j,i)
             IF ( vegetation_pars_f%pars_xy(ind_v_rc_lai,j,i) /= vegetation_pars_f%fill )          &
                surf_lsm%lai(m)           = vegetation_pars_f%pars_xy(ind_v_rc_lai,j,i)
             IF ( vegetation_pars_f%pars_xy(ind_v_c_veg,j,i) /=  vegetation_pars_f%fill )          &
                surf_lsm%c_veg(m)         = vegetation_pars_f%pars_xy(ind_v_c_veg,j,i)
             IF ( vegetation_pars_f%pars_xy(ind_v_gd,j,i) /= vegetation_pars_f%fill )              &
                surf_lsm%g_d(m)           = vegetation_pars_f%pars_xy(ind_v_gd,j,i)
             IF ( vegetation_pars_f%pars_xy(ind_v_z0,j,i) /= vegetation_pars_f%fill )              &
                surf_lsm%z0(m)            = vegetation_pars_f%pars_xy(ind_v_z0,j,i)
             IF ( vegetation_pars_f%pars_xy(ind_v_z0qh,j,i) /= vegetation_pars_f%fill )  THEN
                surf_lsm%z0h(m)           = vegetation_pars_f%pars_xy(ind_v_z0qh,j,i)
                surf_lsm%z0q(m)           = vegetation_pars_f%pars_xy(ind_v_z0qh,j,i)
             ENDIF
             IF ( vegetation_pars_f%pars_xy(ind_v_lambda_s,j,i) /= vegetation_pars_f%fill )        &
                surf_lsm%lambda_surface_s(m) = vegetation_pars_f%pars_xy(ind_v_lambda_s,j,i)
             IF ( vegetation_pars_f%pars_xy(ind_v_lambda_u,j,i) /= vegetation_pars_f%fill )        &
                surf_lsm%lambda_surface_u(m) = vegetation_pars_f%pars_xy(ind_v_lambda_u,j,i)
             IF ( vegetation_pars_f%pars_xy(ind_v_f_sw_in,j,i) /= vegetation_pars_f%fill )         &
                surf_lsm%f_sw_in(m)          = vegetation_pars_f%pars_xy(ind_v_f_sw_in,j,i)
             IF ( vegetation_pars_f%pars_xy(ind_v_c_surf,j,i) /= vegetation_pars_f%fill )          &
                surf_lsm%c_surface(m)        = vegetation_pars_f%pars_xy(ind_v_c_surf,j,i)
             IF ( vegetation_pars_f%pars_xy(ind_v_at,j,i) /= vegetation_pars_f%fill )              &
                surf_lsm%albedo_type(m,ind_veg_wall) = INT( vegetation_pars_f%pars_xy(ind_v_at,j,i) )
             IF ( vegetation_pars_f%pars_xy(ind_v_emis,j,i) /= vegetation_pars_f%fill )            &
                surf_lsm%emissivity(m,ind_veg_wall)  = vegetation_pars_f%pars_xy(ind_v_emis,j,i)
          ENDIF
       ENDDO

    ENDIF

!
!-- Level 1, initialization of water parameters. A horizontally homogeneous distribution is assumed
!-- here.
    IF ( water_type /= 0 )  THEN

       IF ( water_temperature == 9999999.9_wp )  THEN
          water_temperature = water_pars(ind_w_temp,water_type)
       ENDIF

       IF ( z0_water == 9999999.9_wp )  THEN
          z0_water = water_pars(ind_w_z0,water_type)
       ENDIF

       IF ( z0h_water == 9999999.9_wp )  THEN
          z0h_water = water_pars(ind_w_z0h,water_type)
       ENDIF

       IF ( z0q_water == 9999999.9_wp )  THEN
          z0q_water = water_pars(ind_w_z0h,water_type)
       ENDIF

       IF ( albedo_type == 9999999  .AND.  albedo == 9999999.9_wp )  THEN
          albedo_type = INT(water_pars(ind_w_at,water_type))
       ENDIF

       IF ( emissivity == 9999999.9_wp )  THEN
          emissivity = water_pars(ind_w_emis,water_type)
       ENDIF

    ENDIF
!
!-- Map values onto surfaces
    DO  m = 1, surf_lsm%ns
       IF ( surf_lsm%water_surface(m) )  THEN
          IF ( TRIM( initializing_actions ) /= 'read_restart_data'  .AND.                          &
               .NOT. read_spinup_data  )  THEN
             t_soil%var_2d(:,m)        = water_temperature
          ENDIF
          surf_lsm%z0(m)               = z0_water
          surf_lsm%z0h(m)              = z0h_water
          surf_lsm%z0q(m)              = z0q_water
          surf_lsm%lambda_surface_s(m) = 1.0E10_wp
          surf_lsm%lambda_surface_u(m) = 1.0E10_wp
          surf_lsm%c_surface(m)        = 0.0_wp
          surf_lsm%albedo_type(m,ind_wat_win) = albedo_type
          surf_lsm%emissivity(m,ind_wat_win)  = emissivity

          surf_lsm%water_type(m)      = water_type
          surf_lsm%water_type_name(m) = water_type_name(water_type)
       ENDIF
    ENDDO
!
!-- Level 2, initialization of water parameters via water_type read from file. Water surfaces are
!-- initialized for each (y,x)-grid point individually using default paramter settings according to
!-- the given water type.
    IF ( water_type_f%from_file )  THEN
       DO  m = 1, surf_lsm%ns
          i = surf_lsm%i(m) + MERGE( 0, surf_lsm%ioff(m), surf_lsm%building_covered(m) )
          j = surf_lsm%j(m) + MERGE( 0, surf_lsm%joff(m), surf_lsm%building_covered(m) )

          st = water_type_f%var(j,i)
          IF ( st /= water_type_f%fill  .AND.  st /= 0 )  THEN
             IF ( TRIM( initializing_actions ) /= 'read_restart_data'  .AND.                       &
                  .NOT. read_spinup_data )                                                         &
             THEN
                t_soil%var_2d(:,m) = water_pars(ind_w_temp,st)
             ENDIF
             surf_lsm%z0(m)     = water_pars(ind_w_z0,st)
             surf_lsm%z0h(m)    = water_pars(ind_w_z0h,st)
             surf_lsm%z0q(m)    = water_pars(ind_w_z0h,st)
             surf_lsm%lambda_surface_s(m) = water_pars(ind_w_lambda_s,st)
             surf_lsm%lambda_surface_u(m) = water_pars(ind_w_lambda_u,st)
             surf_lsm%c_surface(m)        = 0.0_wp
             surf_lsm%albedo_type(m,ind_wat_win) = INT( water_pars(ind_w_at,st) )
             surf_lsm%emissivity(m,ind_wat_win)  = water_pars(ind_w_emis,st)

             surf_lsm%water_type(m)      = st
             surf_lsm%water_type_name(m) = water_type_name(st)
          ENDIF
       ENDDO
    ENDIF
!
!-- Level 3, initialization of water parameters at single (x,y) position via water_pars read from
!-- file.
    IF ( water_pars_f%from_file )  THEN
       DO  m = 1, surf_lsm%ns
          i = surf_lsm%i(m) + MERGE( 0, surf_lsm%ioff(m), surf_lsm%building_covered(m) )
          j = surf_lsm%j(m) + MERGE( 0, surf_lsm%joff(m), surf_lsm%building_covered(m) )
!
!--       If surface element is not a water surface and any value in water_pars is given, neglect
!--       this information and give an informative message that this value will not be used.
          IF ( .NOT. surf_lsm%water_surface(m)  .AND.                                              &
                ANY( water_pars_f%pars_xy(:,j,i) /= water_pars_f%fill ) )                          &
          THEN
             WRITE( message_string, * ) 'surface element at grid point (i,j) = (',                 &
                                        i, j, ') is not a water surface, ',                        &
                                        'so that information given in ',                           &
                                        'water_pars at this point is neglected.'
             CALL message( 'land_surface_model_mod', 'LSM0042', 0, 0, myid, 6, 0 )
          ELSE
             IF ( water_pars_f%pars_xy(ind_w_temp,j,i) /= water_pars_f%fill  .AND.                 &
                  TRIM( initializing_actions ) /= 'read_restart_data'  .AND.                       &
                  .NOT. read_spinup_data )                                                         &
          THEN
                t_soil%var_2d(:,m) = water_pars_f%pars_xy(ind_w_temp,j,i)
             ENDIF

             IF ( water_pars_f%pars_xy(ind_w_z0,j,i) /= water_pars_f%fill )                        &
                surf_lsm%z0(m)     = water_pars_f%pars_xy(ind_w_z0,j,i)

             IF ( water_pars_f%pars_xy(ind_w_z0h,j,i) /= water_pars_f%fill )  THEN
                surf_lsm%z0h(m)    = water_pars_f%pars_xy(ind_w_z0h,j,i)
                surf_lsm%z0q(m)    = water_pars_f%pars_xy(ind_w_z0h,j,i)
             ENDIF
             IF ( water_pars_f%pars_xy(ind_w_lambda_s,j,i) /= water_pars_f%fill )                  &
                surf_lsm%lambda_surface_s(m) = water_pars_f%pars_xy(ind_w_lambda_s,j,i)

             IF ( water_pars_f%pars_xy(ind_w_lambda_u,j,i) /= water_pars_f%fill )                  &
                surf_lsm%lambda_surface_u(m) = water_pars_f%pars_xy(ind_w_lambda_u,j,i)

             IF ( water_pars_f%pars_xy(ind_w_at,j,i) /= water_pars_f%fill )                        &
                surf_lsm%albedo_type(m,ind_wat_win) = INT( water_pars_f%pars_xy(ind_w_at,j,i) )

             IF ( water_pars_f%pars_xy(ind_w_emis,j,i) /= water_pars_f%fill )                      &
                surf_lsm%emissivity(m,ind_wat_win) = water_pars_f%pars_xy(ind_w_emis,j,i)
          ENDIF
       ENDDO
    ENDIF
!
!-- Initialize pavement-type surfaces, level 1
    IF ( pavement_type /= 0 )  THEN

!
!--    When a pavement_type is used, overwrite a possible setting of the pavement depth as it is
!--    already defined by the pavement type.
       pavement_depth_level = 0

       IF ( z0_pavement == 9999999.9_wp )  THEN
          z0_pavement  = pavement_pars(ind_p_z0,pavement_type)
       ENDIF

       IF ( z0h_pavement == 9999999.9_wp )  THEN
          z0h_pavement = pavement_pars(ind_p_z0h,pavement_type)
       ENDIF

       IF ( z0q_pavement == 9999999.9_wp )  THEN
          z0q_pavement = pavement_pars(ind_p_z0h,pavement_type)
       ENDIF

       IF ( pavement_heat_conduct == 9999999.9_wp )  THEN
          pavement_heat_conduct = pavement_subsurface_pars_1(0,pavement_type)
       ENDIF

       IF ( pavement_heat_capacity == 9999999.9_wp )  THEN
          pavement_heat_capacity = pavement_subsurface_pars_2(0,pavement_type)
       ENDIF

       IF ( albedo_type == 9999999  .AND.  albedo == 9999999.9_wp )  THEN
          albedo_type = INT(pavement_pars(ind_p_at,pavement_type))
       ENDIF

       IF ( emissivity == 9999999.9_wp )  THEN
          emissivity = pavement_pars(ind_p_emis,pavement_type)
       ENDIF

!
!--    If the depth level of the pavement is not set, determine it from lookup table.
       IF ( pavement_depth_level == 0 )  THEN
          DO  k = nzb_soil, nzt_soil
             IF ( pavement_subsurface_pars_1(k,pavement_type) == 9999999.9_wp  .OR.                &
                  pavement_subsurface_pars_2(k,pavement_type) == 9999999.9_wp )  THEN
                nzt_pavement = k-1
                EXIT
             ENDIF
          ENDDO
       ELSE
          nzt_pavement = pavement_depth_level
       ENDIF
    ENDIF
!
!-- Level 1 initialization of pavement type surfaces. Horizontally homogeneous characteristics are
!-- assumed.
    surf_lsm%nzt_pavement = pavement_depth_level
    DO  m = 1, surf_lsm%ns
       IF ( surf_lsm%pavement_surface(m) )  THEN
          surf_lsm%nzt_pavement(m)        = nzt_pavement
          surf_lsm%z0(m)                  = z0_pavement
          surf_lsm%z0h(m)                 = z0h_pavement
          surf_lsm%z0q(m)                 = z0q_pavement
          surf_lsm%lambda_surface_s(m)    = pavement_heat_conduct * ddz_soil(nzb_soil) * 2.0_wp
          surf_lsm%lambda_surface_u(m)    = pavement_heat_conduct * ddz_soil(nzb_soil) * 2.0_wp
          surf_lsm%c_surface(m)           = pavement_heat_capacity * dz_soil(nzb_soil) * 0.25_wp

          surf_lsm%albedo_type(m,ind_pav_green) = albedo_type
          surf_lsm%emissivity(m,ind_pav_green)  = emissivity

          surf_lsm%pavement_type(m)      = pavement_type
          surf_lsm%pavement_type_name(m) = pavement_type_name(pavement_type)

          IF ( pavement_type /= 0 )  THEN
             DO  k = nzb_soil, surf_lsm%nzt_pavement(m)
                surf_lsm%lambda_h_def(k,m)    = pavement_subsurface_pars_1(k,pavement_type)
                surf_lsm%rho_c_total_def(k,m) = pavement_subsurface_pars_2(k,pavement_type)
             ENDDO
          ELSE
             surf_lsm%lambda_h_def(:,m)     = pavement_heat_conduct
             surf_lsm%rho_c_total_def(:,m)  = pavement_heat_capacity
          ENDIF
       ENDIF
    ENDDO
!
!-- Level 2 initialization of pavement type surfaces via pavement_type read from file. Pavement
!-- surfaces are initialized for each (y,x)-grid point individually.
    IF ( pavement_type_f%from_file )  THEN
       DO  m = 1, surf_lsm%ns
          i = surf_lsm%i(m) + MERGE( 0, surf_lsm%ioff(m), surf_lsm%building_covered(m) )
          j = surf_lsm%j(m) + MERGE( 0, surf_lsm%joff(m), surf_lsm%building_covered(m) )

          st = pavement_type_f%var(j,i)
          IF ( st /= pavement_type_f%fill  .AND.  st /= 0 )  THEN
!
!--          Determine deepmost index of pavement layer
             DO  k = nzb_soil, nzt_soil
                IF ( pavement_subsurface_pars_1(k,st) == 9999999.9_wp  .OR.                        &
                     pavement_subsurface_pars_2(k,st) == 9999999.9_wp )                            &
                THEN
                   surf_lsm%nzt_pavement(m) = k-1
                   EXIT
                ENDIF
             ENDDO

             surf_lsm%z0(m)               = pavement_pars(ind_p_z0,st)
             surf_lsm%z0h(m)              = pavement_pars(ind_p_z0h,st)
             surf_lsm%z0q(m)              = pavement_pars(ind_p_z0h,st)

             surf_lsm%lambda_surface_s(m) = pavement_subsurface_pars_1(0,st)                       &
                                            * ddz_soil(nzb_soil) * 2.0_wp
             surf_lsm%lambda_surface_u(m) = pavement_subsurface_pars_1(0,st)                       &
                                            * ddz_soil(nzb_soil) * 2.0_wp
             surf_lsm%c_surface(m)        = pavement_subsurface_pars_2(0,st)                       &
                                            * dz_soil(nzb_soil)  * 0.25_wp
             surf_lsm%albedo_type(m,ind_pav_green) = INT( pavement_pars(ind_p_at,st) )
             surf_lsm%emissivity(m,ind_pav_green)  = pavement_pars(ind_p_emis,st)

             surf_lsm%pavement_type(m)      = st
             surf_lsm%pavement_type_name(m) = pavement_type_name(st)

             DO  k = nzb_soil, surf_lsm%nzt_pavement(m)
                surf_lsm%lambda_h_def(k,m)    = pavement_subsurface_pars_1(k,pavement_type)
                surf_lsm%rho_c_total_def(k,m) = pavement_subsurface_pars_2(k,pavement_type)
             ENDDO
          ENDIF
       ENDDO
    ENDIF
!
!-- Level 3, initialization of pavement parameters at single (x,y) position via pavement_pars read
!-- from file.
    IF ( pavement_pars_f%from_file )  THEN
       DO  m = 1, surf_lsm%ns
          i = surf_lsm%i(m) + MERGE( 0, surf_lsm%ioff(m), surf_lsm%building_covered(m) )
          j = surf_lsm%j(m) + MERGE( 0, surf_lsm%joff(m), surf_lsm%building_covered(m) )
!
!--       If surface element is not a pavement surface and any value in pavement_pars is given,
!--       neglect this information and give an informative message that this value will not be
!--       used.
          IF ( .NOT. surf_lsm%pavement_surface(m)  .AND.                                           &
                ANY( pavement_pars_f%pars_xy(:,j,i) /= pavement_pars_f%fill ) )                    &
          THEN
             WRITE( message_string, * ) 'surface element at grid point (i,j) = (',                 &
                                        i, j, ') is not a pavement surface, ',                     &
                                        'so that information given in ',                           &
                                        'pavement_pars at this point is neglected.'
             CALL message( 'land_surface_model_mod', 'LSM0043', 0, 0, myid, 6, 0 )
          ELSE
             IF ( pavement_pars_f%pars_xy(ind_p_z0,j,i) /= pavement_pars_f%fill )                  &
                surf_lsm%z0(m)  = pavement_pars_f%pars_xy(ind_p_z0,j,i)
             IF ( pavement_pars_f%pars_xy(ind_p_z0h,j,i) /= pavement_pars_f%fill )  THEN
                surf_lsm%z0h(m) = pavement_pars_f%pars_xy(ind_p_z0h,j,i)
                surf_lsm%z0q(m) = pavement_pars_f%pars_xy(ind_p_z0h,j,i)
             ENDIF
             IF ( pavement_pars_f%pars_xy(ind_p_at,j,i) /= pavement_pars_f%fill )                  &
                surf_lsm%albedo_type(m,ind_pav_green) = INT( pavement_pars_f%pars_xy(ind_p_at,j,i) )
             IF ( pavement_pars_f%pars_xy(ind_p_emis,j,i) /= pavement_pars_f%fill )                &
                surf_lsm%emissivity(m,ind_pav_green)  = pavement_pars_f%pars_xy(ind_p_emis,j,i)
          ENDIF
       ENDDO
    ENDIF
!
!-- Moreover, for grid points which are flagged with pavement-type 0 or where
!-- pavement_subsurface_pars_f is provided, soil heat conductivity and capacity are initialized with
!-- parameters given in pavement_subsurface_pars read from file.
    IF ( pavement_subsurface_pars_f%from_file )  THEN
!
!--    Set pavement depth to nzt_soil. Please note, this is just a workaround at the moment.
       DO  m = 1, surf_lsm%ns
          IF ( surf_lsm%pavement_surface(m) )  THEN

             i = surf_lsm%i(m) + MERGE( 0, surf_lsm%ioff(m), surf_lsm%building_covered(m) )
             j = surf_lsm%j(m) + MERGE( 0, surf_lsm%joff(m), surf_lsm%building_covered(m) )

             surf_lsm%nzt_pavement(m) = nzt_soil

             IF ( pavement_subsurface_pars_f%pars_xyz(ind_p_lambda_h,0,j,i)                        &
                  /= pavement_subsurface_pars_f%fill )                                             &
             THEN
                surf_lsm%lambda_surface_s(m)  =                                                    &
                               pavement_subsurface_pars_f%pars_xyz(ind_p_lambda_h,0,j,i)           &
                               * ddz_soil(nzb_soil) * 2.0_wp
                surf_lsm%lambda_surface_u(m)  =                                                    &
                                    pavement_subsurface_pars_f%pars_xyz(ind_p_lambda_h,0,j,i)      &
                                    * ddz_soil(nzb_soil) * 2.0_wp
             ENDIF
             IF ( pavement_subsurface_pars_f%pars_xyz(ind_p_rho_c,0,j,i) /=                        &
                  pavement_subsurface_pars_f%fill )                                                &
             THEN
                surf_lsm%c_surface(m) = pavement_subsurface_pars_f%pars_xyz(ind_p_rho_c,0,j,i)     &
                                        * dz_soil(nzb_soil) * 0.25_wp
             ENDIF

             DO  k = nzb_soil, nzt_soil
                IF ( pavement_subsurface_pars_f%pars_xyz(ind_p_lambda_h,k,j,i) /=                  &
                     pavement_subsurface_pars_f%fill )                                             &
                THEN
                   surf_lsm%lambda_h_def(k,m) =                                                    &
                                         pavement_subsurface_pars_f%pars_xyz(ind_p_lambda_h,k,j,i)
                ENDIF
                IF ( pavement_subsurface_pars_f%pars_xyz(ind_p_rho_c,k,j,i) /=                     &
                     pavement_subsurface_pars_f%fill )                                             &
                THEN
                   surf_lsm%rho_c_total_def(k,m) =                                                 &
                                         pavement_subsurface_pars_f%pars_xyz(ind_p_rho_c,k,j,i)
                ENDIF
                IF ( ( pavement_subsurface_pars_f%pars_xyz(ind_p_rho_c,k,j,i) /=                   &
                       pavement_subsurface_pars_f%fill ) .AND.                                     &
                     ( pavement_subsurface_pars_f%pars_xyz(ind_p_lambda_h,k,j,i) /=                &
                       pavement_subsurface_pars_f%fill ) )                                         &
                THEN
                   surf_lsm%nzt_pavement(m) = k
                ENDIF
             ENDDO

          ENDIF
       ENDDO
    ENDIF
!
!-- Initialize albedo type via given type from static input file. Please note, even though the
!-- albedo type has been already given by the pars, albedo_type overwrites these values.
    IF ( albedo_type_f%from_file )  THEN
       DO  m = 1, surf_lsm%ns
          i = surf_lsm%i(m) + MERGE( 0, surf_lsm%ioff(m), surf_lsm%building_covered(m) )
          j = surf_lsm%j(m) + MERGE( 0, surf_lsm%joff(m), surf_lsm%building_covered(m) )
          IF ( albedo_type_f%var(j,i) /= albedo_type_f%fill )                                      &
                                             surf_lsm%albedo_type(m,:) = albedo_type_f%var(j,i)
       ENDDO
    ENDIF
!
!-- Initial run actions
    IF ( TRIM( initializing_actions ) /= 'read_restart_data'  .AND.  .NOT. read_spinup_data )  THEN
!
!--    First, initialize soil temperature and moisture.
!--    According to the initialization for surface and soil parameters, initialize soil moisture and
!--    temperature via a level approach. This is to assure that all surface elements are
!--    initialized, even if data provided from input file contains fill values at some locations.
!--    Level 1, initialization via profiles given in parameter file
       DO  m = 1, surf_lsm%ns
          IF ( surf_lsm%vegetation_surface(m)  .OR.  surf_lsm%pavement_surface(m) )  THEN
             DO  k = nzb_soil, nzt_soil
                t_soil%var_2d(k,m) = soil_temperature(k)
                m_soil%var_2d(k,m) = soil_moisture(k)
             ENDDO
             t_soil%var_2d(nzt_soil+1,m) = deep_soil_temperature
          ENDIF
       ENDDO
!
!--    Level 2 initialization of the soil, read soil properties from dynamic input file.
       IF ( input_pids_dynamic )  THEN
!
!--       CPU measurement
          CALL cpu_log( log_point_s(85), 'NetCDF input init', 'start' )
#if defined ( __netcdf )
!
!--       Open file in read-only mode
          CALL open_read_file( TRIM( input_file_dynamic ) // TRIM( coupling_char ), pids_id )
!
!--       Inquire all variable names
          CALL inquire_num_variables( pids_id, num_var_pids )
!
!--       Allocate memory to store variable names.
          ALLOCATE( vars_pids(1:num_var_pids) )
          CALL inquire_variable_names( pids_id, vars_pids )
!
!--       Read vertical dimension for soil depth.
          IF ( check_existence( vars_pids, 'zsoil' ) )                                             &
             CALL get_dimension_length( pids_id, init_3d%nzs, 'zsoil' )
!
!--       Read also the horizontal dimensions required for soil initialization.
!--       Please note, in case of non-nested runs or in case of root domain, these data is already
!--       available, but will be read again for the sake of clearness. Also, check for correct
!--       horizontal dimensions.
          IF ( check_existence( vars_pids, 'x' ) )  THEN
             CALL get_dimension_length( pids_id, init_3d%nx, 'x'  )
             IF ( init_3d%nx-1 /= nx )  THEN
                WRITE( message_string, '(A,I5,A,I5,A)' )                                           &
                           'number of grid points along x in dynamic input file (=', init_3d%nx-1, &
                           ') does not match the number grid points in this run (nx=', nx, ')'
                CALL message( 'lsm_init', 'LSM0044', 1, 2, 0, 6, 0 )
             ENDIF
          ENDIF
          IF ( check_existence( vars_pids, 'y' ) )  THEN
             CALL get_dimension_length( pids_id, init_3d%ny, 'y'  )
             IF ( init_3d%ny-1 /= ny )  THEN
                WRITE( message_string, '(A,I5,A,I5,A)' )                                           &
                           'number of grid points along y in dynamic input file (=', init_3d%ny-1, &
                           ') does not match the number grid points in this run (ny=', ny, ')'
                CALL message( 'lsm_init', 'LSM0044', 1, 2, 0, 6, 0 )
             ENDIF
          ENDIF
!
!--       Read vertical dimensions. Later, these are required for eventual inter- and extrapolations
!--       of the initialization data.
          IF ( check_existence( vars_pids, 'zsoil' ) )  THEN
             ALLOCATE( init_3d%z_soil(1:init_3d%nzs) )
             CALL get_variable( pids_id, 'zsoil', init_3d%z_soil )
          ENDIF
!
!--       Read initial data for soil moisture
          IF ( check_existence( vars_pids, 'init_soil_m' ) )  THEN
!
!--          Read attributes for the fill value and level-of-detail
             CALL get_attribute( pids_id, char_fill, init_3d%fill_msoil, .FALSE., 'init_soil_m' )
             CALL get_attribute( pids_id, char_lod, init_3d%lod_msoil, .FALSE., 'init_soil_m' )
!
!--          level-of-detail 1 - read initialization profile
             IF ( init_3d%lod_msoil == 1 )  THEN
                ALLOCATE( init_3d%msoil_1d(0:init_3d%nzs-1) )

                CALL get_variable( pids_id, 'init_soil_m', init_3d%msoil_1d(0:init_3d%nzs-1) )
!
!--          level-of-detail 2 - read 3D initialization data
             ELSEIF ( init_3d%lod_msoil == 2 )  THEN
                ALLOCATE ( init_3d%msoil_3d(0:init_3d%nzs-1,nys:nyn,nxl:nxr) )

               CALL get_variable( pids_id, 'init_soil_m',                                          &
                                  init_3d%msoil_3d(0:init_3d%nzs-1,nys:nyn,nxl:nxr),               &
                                  nxl, nxr, nys, nyn, 0, init_3d%nzs-1 )

             ENDIF
             init_3d%from_file_msoil = .TRUE.
          ENDIF
!
!--       Read soil temperature
          IF ( check_existence( vars_pids, 'init_soil_t' ) )  THEN
!
!--          Read attributes for the fill value and level-of-detail
             CALL get_attribute( pids_id, char_fill, init_3d%fill_tsoil, .FALSE., 'init_soil_t' )
             CALL get_attribute( pids_id, char_lod, init_3d%lod_tsoil, .FALSE., 'init_soil_t' )
!
!--          level-of-detail 1 - read initialization profile
             IF ( init_3d%lod_tsoil == 1 )  THEN
                ALLOCATE( init_3d%tsoil_1d(0:init_3d%nzs-1) )

                CALL get_variable( pids_id, 'init_soil_t', init_3d%tsoil_1d(0:init_3d%nzs-1) )

!
!--          level-of-detail 2 - read 3D initialization data
             ELSEIF ( init_3d%lod_tsoil == 2 )  THEN
                ALLOCATE ( init_3d%tsoil_3d(0:init_3d%nzs-1,nys:nyn,nxl:nxr) )

                CALL get_variable( pids_id, 'init_soil_t',                                         &
                                   init_3d%tsoil_3d(0:init_3d%nzs-1,nys:nyn,nxl:nxr),              &
                                   nxl, nxr, nys, nyn, 0, init_3d%nzs-1 )
             ENDIF
             init_3d%from_file_tsoil = .TRUE.
          ENDIF
!
!--       Close the input file and deallocate temporary arrays
          DEALLOCATE( vars_pids )

          CALL close_input_file( pids_id )
#endif
!
!--       End of CPU measurement
          CALL cpu_log( log_point_s(85), 'NetCDF input init', 'stop' )
       ENDIF
!
!--    In case no dynamic input is available for a child domain but the parent is initialized with
!--    dynamic input file, the different soil states can lead to significant discrepancies in the
!--    atmospheric surface forcing. For this reason, the child domain is initialized with
!--    domain-averaged soil profiles from the root domain, even if no initialization with external
!--    data is set. Note, as long as a dynamic input file with soil information is available for
!--    the child domain, the input file information will be used.
       IF ( nested_run )  THEN
#if defined( __parallel )
!
!--       Check if soil moisture and temperature in the root model are initialized from dynamic
!--       input. This case, distribute these information to its child domain(s).
          IF ( root_model )  THEN
             init_msoil_from_driver_root = init_3d%from_file_msoil
             init_tsoil_from_driver_root = init_3d%from_file_tsoil
          ENDIF

          CALL MPI_BCAST( init_msoil_from_driver_root, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD, ierr )
          CALL MPI_BCAST( init_tsoil_from_driver_root, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD, ierr )
!
!--       In case of a nested run, first average the soil profiles in the root domain.
          IF ( init_msoil_from_driver_root  .OR.  init_tsoil_from_driver_root )  THEN

             IF ( root_model )  THEN
!
!--             Child domains will be only initialized with horizontally averaged soil profiles in
!--             parent domain (for sake of simplicity).
!--             If required, average soil data on root parent domain before the soil profiles are
!--             distributed onto the child domains.
!--             Start with soil moisture.
                IF ( init_3d%from_file_msoil  .AND.  init_3d%lod_msoil == 2 )  THEN
                   ALLOCATE( pr_soil_init(0:init_3d%nzs-1) )
                   DO  k = 0, init_3d%nzs-1
                      pr_soil_init(k) = SUM( init_3d%msoil_3d(k,nys:nyn,nxl:nxr)  )
                   ENDDO
!
!--                Allocate 1D array for soil-moisture profile (will not be allocated in lod==2
!--                case).
                   ALLOCATE( init_3d%msoil_1d(0:init_3d%nzs-1) )
                   init_3d%msoil_1d = 0.0_wp
                   CALL MPI_ALLREDUCE( pr_soil_init(0), init_3d%msoil_1d(0), SIZE(pr_soil_init),   &
                                       MPI_REAL, MPI_SUM, comm2d, ierr )

                   init_3d%msoil_1d = init_3d%msoil_1d / REAL( ( nx + 1 ) * ( ny + 1), KIND=wp )
                   DEALLOCATE( pr_soil_init )
                ENDIF
!
!--             Proceed with soil temperature.
                IF ( init_3d%from_file_tsoil  .AND.  init_3d%lod_tsoil == 2 )  THEN
                   ALLOCATE( pr_soil_init(0:init_3d%nzs-1) )

                   DO  k = 0, init_3d%nzs-1
                      pr_soil_init(k) = SUM( init_3d%tsoil_3d(k,nys:nyn,nxl:nxr)  )
                   ENDDO
!
!--                Allocate 1D array for soil-temperature profile (will not be allocated in lod==2
!--                case).
                   ALLOCATE( init_3d%tsoil_1d(0:init_3d%nzs-1) )
                   init_3d%tsoil_1d = 0.0_wp
                   CALL MPI_ALLREDUCE( pr_soil_init(0), init_3d%tsoil_1d(0), SIZE(pr_soil_init),   &
                                       MPI_REAL, MPI_SUM, comm2d, ierr )
                   init_3d%tsoil_1d = init_3d%tsoil_1d / REAL( ( nx + 1 ) * ( ny + 1), KIND=wp )
                   DEALLOCATE( pr_soil_init )

                ENDIF
             ENDIF
!
!--          Broadcast number of soil layers in root model to all childs.
!--          Note, only process 0 in COMM_WORLD is sending.
             IF ( root_model )  nzs_root = init_3d%nzs

             CALL MPI_BCAST( nzs_root, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr )
!
!--          Allocate dummy arrays for soil moisture and temperature profiles on all domains.
             ALLOCATE( z_soil_root(1:nzs_root)   )
             IF ( init_msoil_from_driver_root )  ALLOCATE( m_soil_root(0:nzs_root-1) )
             IF ( init_tsoil_from_driver_root )  ALLOCATE( t_soil_root(0:nzs_root-1) )
!
!--          Distribute the mean soil profiles to all child domains.
             IF ( root_model )  THEN
                z_soil_root = init_3d%z_soil
                IF ( init_msoil_from_driver_root )  m_soil_root = init_3d%msoil_1d
                IF ( init_tsoil_from_driver_root )  t_soil_root = init_3d%tsoil_1d
             ENDIF

             CALL MPI_BCAST( z_soil_root, SIZE( z_soil_root ), MPI_REAL, 0, MPI_COMM_WORLD, ierr )

             IF ( init_msoil_from_driver_root )                                                    &
                CALL MPI_BCAST( m_soil_root, SIZE( m_soil_root ), MPI_REAL, 0, MPI_COMM_WORLD,     &
                                ierr )

             IF ( init_msoil_from_driver_root )                                                    &
                CALL MPI_BCAST( t_soil_root, SIZE( t_soil_root ), MPI_REAL, 0, MPI_COMM_WORLD,     &
                                ierr )
!
!--          In the following, the child domains decide whether they take the information from the
!--          root domain or not.
             IF ( .NOT. root_model )  THEN
!
!--             If soil moisture or temperature isn't in dynamic input file for the child, take the
!--             information provided from the root model.
!--             Start with z-dimension
                IF ( .NOT. init_3d%from_file_msoil  .OR.  .NOT. init_3d%from_file_msoil )  THEN
                   init_3d%nzs = nzs_root
                   ALLOCATE( init_3d%z_soil(1:init_3d%nzs) )
                   init_3d%z_soil(1:init_3d%nzs) = z_soil_root
                ENDIF
!
!--             Take soil moisture. Note, control flags from_file... and LoD need to be set.
                IF ( .NOT. init_3d%from_file_msoil )  THEN
                   ALLOCATE( init_3d%msoil_1d(0:init_3d%nzs-1) )
                   init_3d%lod_msoil = 1
                   init_3d%from_file_msoil = .TRUE.

                   init_3d%msoil_1d = m_soil_root
                ENDIF
!
!--             Take soil temperature. Note, control flags from_file... and LoD need to be set.
                IF ( .NOT. init_3d%from_file_tsoil )  THEN
                   ALLOCATE( init_3d%tsoil_1d(0:init_3d%nzs-1) )
                   init_3d%lod_tsoil = 1
                   init_3d%from_file_tsoil = .TRUE.

                   init_3d%tsoil_1d = t_soil_root
                ENDIF
             ENDIF

             DEALLOCATE( z_soil_root )
             DEALLOCATE( m_soil_root )
             DEALLOCATE( t_soil_root )
          ENDIF
#endif
       ENDIF
!
!--    Proceed with Level 2 initialization.
       IF ( init_3d%from_file_msoil )  THEN

          IF ( init_3d%lod_msoil == 1 )  THEN
             DO  m = 1, surf_lsm%ns
                IF ( surf_lsm%vegetation_surface(m)  .OR.  surf_lsm%pavement_surface(m) )  THEN
                   CALL interpolate_soil_profile( m_soil%var_2d(nzb_soil:nzt_soil,m),              &
                                                  init_3d%msoil_1d(:),                             &
                                                  zs(nzb_soil:nzt_soil), init_3d%z_soil,           &
                                                  nzb_soil, nzt_soil, nzb_soil, init_3d%nzs-1 )
                ENDIF
             ENDDO
          ELSE
             DO  m = 1, surf_lsm%ns
                IF ( surf_lsm%vegetation_surface(m)  .OR.  surf_lsm%pavement_surface(m) )  THEN
                   i = surf_lsm%i(m)
                   j = surf_lsm%j(m)

                   IF ( init_3d%msoil_3d(0,j,i) /= init_3d%fill_msoil )                            &
                      CALL interpolate_soil_profile( m_soil%var_2d(nzb_soil:nzt_soil,m),           &
                                                     init_3d%msoil_3d(:,j,i),                      &
                                                     zs(nzb_soil:nzt_soil), init_3d%z_soil,        &
                                                     nzb_soil, nzt_soil, nzb_soil, init_3d%nzs-1 )
                ENDIF
             ENDDO
          ENDIF
       ENDIF
!
!--    Soil temperature
       IF ( init_3d%from_file_tsoil )  THEN

          IF ( init_3d%lod_tsoil == 1 )  THEN
             DO  m = 1, surf_lsm%ns
                IF ( surf_lsm%vegetation_surface(m)  .OR.  surf_lsm%pavement_surface(m) )  THEN
                   CALL interpolate_soil_profile( t_soil%var_2d(nzb_soil:nzt_soil,m),              &
                                                  init_3d%tsoil_1d(:),                             &
                                                  zs(nzb_soil:nzt_soil), init_3d%z_soil,           &
                                                  nzb_soil, nzt_soil, nzb_soil, init_3d%nzs-1 )
!
!--                Set boundary condition, i.e. deep soil temperature
                   t_soil%var_2d(nzt_soil+1,m) = t_soil%var_2d(nzt_soil,m)
                ENDIF
             ENDDO
          ELSE
             DO  m = 1, surf_lsm%ns
                IF ( surf_lsm%vegetation_surface(m)  .OR.  surf_lsm%pavement_surface(m) )  THEN
                   i = surf_lsm%i(m)
                   j = surf_lsm%j(m)

                   IF ( init_3d%tsoil_3d(0,j,i) /= init_3d%fill_tsoil )                            &
                      CALL interpolate_soil_profile( t_soil%var_2d(nzb_soil:nzt_soil,m),           &
                                                     init_3d%tsoil_3d(:,j,i),                      &
                                                     zs(nzb_soil:nzt_soil), init_3d%z_soil,        &
                                                     nzb_soil, nzt_soil, nzb_soil, init_3d%nzs-1 )
!
!--                Set boundary condition, i.e. deep soil temperature
                   t_soil%var_2d(nzt_soil+1,m) = t_soil%var_2d(nzt_soil,m)
                ENDIF
             ENDDO
          ENDIF
       ENDIF
!
!--    After soil moisture and temperature are finally initialized, check if soil moisture is higher
!--    than its saturation value. This case, the soil model parametrization will produce floating
!--    point errors. Hence, limit the soil moisture to its saturation value and give a warning.
!--    Further, check also for negative soil-moisture values.
       flag_exceed_sat = .FALSE.
       flag_negative_value = .FALSE.
       DO  m = 1, surf_lsm%ns
          IF ( surf_lsm%vegetation_surface(m)  .OR.  surf_lsm%pavement_surface(m) )  THEN
             DO  k = nzb_soil, nzt_soil
                IF ( m_soil%var_2d(k,m) > surf_lsm%m_sat(k,m) )  THEN
                   m_soil%var_2d(k,m) = surf_lsm%m_sat(k,m)
                   flag_exceed_sat = .TRUE.
                ENDIF
                IF ( m_soil%var_2d(k,m) < 0.0_wp )  THEN
                   flag_negative_value = .TRUE.
                ENDIF
             ENDDO
          ENDIF
       ENDDO

#if defined( __parallel )
       CALL MPI_ALLREDUCE( MPI_IN_PLACE, flag_exceed_sat, 1, MPI_LOGICAL, MPI_LOR, comm2d, ierr)
#endif
       IF ( flag_exceed_sat )  THEN
          WRITE( message_string, * ) 'Soil moisture is higher than its saturation value ' //       &
                                     'at one or more locations.&Soil moisture is limited to ' //   &
                                     'its saturation value to maintain stability.'
          CALL message( 'lsm_init', 'LSM0045', 0, 1, 0, 6, 0 )
       ENDIF

#if defined( __parallel )
       CALL MPI_ALLREDUCE( MPI_IN_PLACE, flag_negative_value, 1, MPI_LOGICAL, MPI_LOR, comm2d, ierr)
#endif
       IF ( flag_negative_value )  THEN
          WRITE( message_string, * ) 'initial values of soil moisture are < 0.0'
          CALL message( 'lsm_init', 'LSM0046', 1, 2, 0, 6, 0 )
       ENDIF
!
!--    Further initialization
       DO  m = 1, surf_lsm%ns

          i = surf_lsm%i(m)
          j = surf_lsm%j(m)
          k = surf_lsm%k(m)
!
!--       Initialize surface temperature with soil temperature in the uppermost layer.
          t_surface%var_1d(m)    = t_soil%var_2d(nzb_soil,m)
          surf_lsm%pt_surface(m) = t_soil%var_2d(nzb_soil,m) / exner(nzb)

          IF ( bulk_cloud_model  .OR.  cloud_droplets )  THEN
             surf_lsm%pt1(m) = pt(k,j,i) + lv_d_cp * d_exner(k) * ql(k,j,i)
          ELSE
             surf_lsm%pt1(m) = pt(k,j,i)
          ENDIF
!
!--       Assure that r_a cannot be zero at model start
          IF ( surf_lsm%pt1(m) == surf_lsm%pt_surface(m) )                                         &
             surf_lsm%pt1(m) = surf_lsm%pt1(m) + 1.0E-20_wp

          surf_lsm%us(m)   = 0.1_wp
          surf_lsm%ts(m)   = ( surf_lsm%pt1(m) - surf_lsm%pt_surface(m) ) / surf_lsm%r_a(m)
          surf_lsm%shf(m)  = -surf_lsm%us(m) * surf_lsm%ts(m) * rho_surface
       ENDDO

    ENDIF
!
!-- Level 1 initialization of root distribution - provided by the user via via namelist.
    DO  m = 1, surf_lsm%ns
       DO  k = nzb_soil, nzt_soil
          surf_lsm%root_fr(k,m) = root_fraction(k)
       ENDDO
    ENDDO

!
!-- Level 2 initialization of root distribution.
!-- When no root distribution is given by the user, use look-up table to prescribe the root fraction
!-- in the individual soil layers.
    IF ( ALL( root_fraction == 9999999.9_wp ) )  THEN
!
!--    First, calculate the index bounds for integration
       n_soil_layers_total = nzt_soil - nzb_soil + 6
       ALLOCATE ( bound(0:n_soil_layers_total) )
       ALLOCATE ( bound_root_fr(0:n_soil_layers_total) )

       kn = 0
       ko = 0
       bound(0) = 0.0_wp
       DO k = 1, n_soil_layers_total-1
          IF ( zs_layer(kn) <= zs_ref(ko) )  THEN
             bound(k) = zs_layer(kn)
             bound_root_fr(k) = ko
             kn = kn + 1
             IF ( kn > nzt_soil+1 )  THEN
                kn = nzt_soil
             ENDIF
          ELSE
             bound(k) = zs_ref(ko)
             bound_root_fr(k) = ko
             ko = ko + 1
             IF ( ko > 3 )  THEN
                ko = 3
             ENDIF
          ENDIF
       ENDDO
!
!--    Integrate over all soil layers based on the four-layer root fraction.
       kzs = 1
       root_fraction = 0.0_wp
       DO k = 0, n_soil_layers_total-2
          kroot = bound_root_fr(k+1)
          root_fraction(kzs-1) = root_fraction(kzs-1)                                              &
                                 + root_distribution(kroot,vegetation_type)                        &
                                 / dz_soil_ref(kroot) * ( bound(k+1) - bound(k) )

          IF ( bound(k+1) == zs_layer(kzs-1) )  THEN
             kzs = kzs+1
          ENDIF
       ENDDO
!
!--    Normalize so that the sum of all fractions equals one
       root_fraction = root_fraction / SUM(root_fraction)

       DEALLOCATE( bound )
       DEALLOCATE( bound_root_fr )
!
!--    Map calculated root fractions
       DO  m = 1, surf_lsm%ns
          DO  k = nzb_soil, nzt_soil
             IF ( surf_lsm%pavement_surface(m)  .AND.  k <= surf_lsm%nzt_pavement(m) )  THEN
                surf_lsm%root_fr(k,m) = 0.0_wp
             ELSE
                surf_lsm%root_fr(k,m) = root_fraction(k)
             ENDIF

          ENDDO
!
!--       Normalize so that the sum = 1. Only relevant when the root distribution was set to zero
!--       due to pavement at some layers.
          IF ( SUM( surf_lsm%root_fr(:,m) ) > 0.0_wp )  THEN
             DO k = nzb_soil, nzt_soil
                surf_lsm%root_fr(k,m) = surf_lsm%root_fr(k,m) / SUM( surf_lsm%root_fr(:,m) )
             ENDDO
          ENDIF
       ENDDO

    ENDIF
!
!-- Level 3 initialization of root distribution.
!-- Take value from file.
    IF ( root_area_density_lsm_f%from_file )  THEN
       DO  m = 1, surf_lsm%ns
          IF ( surf_lsm%vegetation_surface(m) )  THEN
             i = surf_lsm%i(m)
             j = surf_lsm%j(m)
             DO  k = nzb_soil, nzt_soil
                surf_lsm%root_fr(k,m) = root_area_density_lsm_f%var(k,j,i)
             ENDDO
          ENDIF
       ENDDO
    ENDIF

!
!-- Possibly do user-defined actions (e.g. define heterogeneous land surface)
    CALL user_init_land_surface
!
!-- Calculate new roughness lengths (for water surfaces only, i.e. only horizontal surfaces)
    IF ( .NOT. constant_roughness )  CALL calc_z0_water_surface

    t_soil_p    = t_soil
    m_soil_p    = m_soil
    m_liq_p     = m_liq
    t_surface_p = t_surface

!
!-- Store initial profiles of t_soil and m_soil (assuming they are
!-- horizontally homogeneous on this PE)
!-- DEACTIVATED FOR NOW - leads to error when number of locations with
!-- soil model is zero on a PE.
!        hom(nzb_soil:nzt_soil,1,90,:)  = SPREAD( t_soil%var_2d(nzb_soil:nzt_soil,1),  &
!                                                 2, statistic_regions+1 )
!        hom(nzb_soil:nzt_soil,1,92,:)  = SPREAD( m_soil%var_2d(nzb_soil:nzt_soil,1),  &
!                                                 2, statistic_regions+1 )

!
!-- Finally, make some consistency checks.
!-- Check for illegal combination of LAI and vegetation coverage.
    IF ( ANY( .NOT. surf_lsm%pavement_surface  .AND.                                               &
                    surf_lsm%lai == 0.0_wp     .AND.  surf_lsm%c_veg == 1.0_wp ) )                 &
    THEN
       message_string = 'for non-pavement surfaces the combination ' //                            &
                        ' lai = 0.0 and c_veg = 1.0 is not allowed'
       CALL message( 'lsm_init', 'LSM0047', 2, 2, myid, 6, 0 )
    ENDIF

!
!-- Check if roughness length for momentum, heat, or moisture exceed surface-layer height and
!-- limit local roughness length where necessary (if allowed). If limited, give an informative
!-- message only once in order to avoid the job protocol to be messed-up with messages.
    flag_exceed_z0  = .FALSE.
    flag_exceed_z0h = .FALSE.
    flag_exceed_z0q = .FALSE.
    DO  m = 1, surf_lsm%ns

       IF ( surf_lsm%z0(m) >= 0.5 * surf_lsm%z_mo(m) )  THEN
          flag_exceed_z0 = .TRUE.
          IF ( allow_roughness_limitation )  THEN
             surf_lsm%z0(m) = 0.5_wp * surf_lsm%z_mo(m)
          ELSE
             exit_index = m
             EXIT
          ENDIF
       ENDIF

       IF ( surf_lsm%z0h(m) >= surf_lsm%z_mo(m) )  THEN
          flag_exceed_z0h = .TRUE.
          IF ( allow_roughness_limitation )  THEN
             surf_lsm%z0h(m) = 0.5_wp * surf_lsm%z_mo(m)
          ELSE
             exit_index = m
             EXIT
          ENDIF
       ENDIF

       IF ( surf_lsm%z0q(m) >= surf_lsm%z_mo(m) )  THEN
          flag_exceed_z0q = .TRUE.
          IF ( allow_roughness_limitation )  THEN
             surf_lsm%z0q(m) = 0.5_wp * surf_lsm%z_mo(m)
          ELSE
             exit_index = m
             EXIT
          ENDIF
       ENDIF

    ENDDO

    IF ( flag_exceed_z0  .AND.  .NOT. allow_roughness_limitation )  THEN
       WRITE( message_string, '(A,I6,A,I6,A)' )                                                    &
              'z0 exceeds 0.5 * surface-layer height at natural surface grid point (i,j) = (',     &
               surf_lsm%i(exit_index)+surf_lsm%ioff(exit_index), ',',                              &
               surf_lsm%j(exit_index)+surf_lsm%joff(exit_index), ')'
       CALL message( 'lsm_init', 'LSM0048', 2, 2, myid, 6, 0 )
    ENDIF
#if defined( __parallel )
    CALL MPI_ALLREDUCE( MPI_IN_PLACE, flag_exceed_z0, 1, MPI_LOGICAL, MPI_LOR, comm2d, ierr)
#endif
    IF ( flag_exceed_z0 )  THEN
       WRITE( message_string, * ) 'z0 exceeds 0.5 * surface-layer height at natural surface(s)' // &
                                  ' and is limited to that height'
       CALL message( 'lsm_init', 'LSM0049', 0, 0, 0, 6, 0 )
    ENDIF

    IF ( flag_exceed_z0h  .AND.  .NOT. allow_roughness_limitation )  THEN
       WRITE( message_string, '(A,I6,A,I6,A)' )                                                    &
              'z0h exceeds 0.5 * surface-layer height at natural surface grid point (i,j) = (',    &
               surf_lsm%i(exit_index)+surf_lsm%ioff(exit_index), ',',                              &
               surf_lsm%j(exit_index)+surf_lsm%joff(exit_index), ')'
       CALL message( 'lsm_init', 'LSM0048', 2, 2, myid, 6, 0 )
    ENDIF
#if defined( __parallel )
    CALL MPI_ALLREDUCE( MPI_IN_PLACE, flag_exceed_z0h, 1, MPI_LOGICAL, MPI_LOR, comm2d, ierr)
#endif
    IF ( flag_exceed_z0h )  THEN
       WRITE( message_string, * ) 'z0h exceeds 0.5 * surface-layer height at natural surface(s)'// &
                                  ' and is limited to that height'
       CALL message( 'lsm_init', 'LSM0049', 0, 0, 0, 6, 0 )
    ENDIF

    IF ( flag_exceed_z0q  .AND.  .NOT. allow_roughness_limitation )  THEN
       WRITE( message_string, '(A,I6,A,I6,A)' )                                                    &
              'z0q exceeds 0.5 * surface-layer height at natural surface grid point (i,j) = (',    &
               surf_lsm%i(exit_index)+surf_lsm%ioff(exit_index), ',',                              &
               surf_lsm%j(exit_index)+surf_lsm%joff(exit_index), ')'
       CALL message( 'lsm_init', 'LSM0048', 2, 2, myid, 6, 0 )
    ENDIF
#if defined( __parallel )
    CALL MPI_ALLREDUCE( MPI_IN_PLACE, flag_exceed_z0q, 1, MPI_LOGICAL, MPI_LOR, comm2d, ierr)
#endif
    IF ( flag_exceed_z0q )  THEN
       WRITE( message_string, * ) 'z0q exceeds 0.5 * surface-layer height at natural surface(s)'// &
                                  ' and is limited to that height'
       CALL message( 'lsm_init', 'LSM0049', 0, 0, 0, 6, 0 )
    ENDIF

!
!-- Finally, set maximum allowed timestep to a huge number, so that it does not affect the
!-- timestep calculation at the very first call.
    surf_lsm%dt_max = HUGE( 1.0_wp )

    IF ( debug_output )  CALL debug_message( 'lsm_init', 'end' )

 END SUBROUTINE lsm_init


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Allocate land surface model arrays and define pointers
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_init_arrays

    ALLOCATE( root_extr(nzb_soil:nzt_soil) )
    root_extr = 0.0_wp
!
!-- Allocate surface and soil temperature / humidity. Please note, these arrays are allocated
!-- according to surface-data structure, even if they do not belong to the data type due to the
!-- pointer arithmetric (TARGET attribute is not allowed in a data-type).
    ALLOCATE( m_liq_1%var_1d(1:surf_lsm%ns) )
    ALLOCATE( m_liq_2%var_1d(1:surf_lsm%ns) )
    ALLOCATE( t_surface_1%var_1d(1:surf_lsm%ns) )
    ALLOCATE( t_surface_2%var_1d(1:surf_lsm%ns) )
    ALLOCATE( m_soil_1%var_2d(nzb_soil:nzt_soil,1:surf_lsm%ns) )
    ALLOCATE( m_soil_2%var_2d(nzb_soil:nzt_soil,1:surf_lsm%ns) )
    ALLOCATE( t_soil_1%var_2d(nzb_soil:nzt_soil+1,1:surf_lsm%ns) )
    ALLOCATE( t_soil_2%var_2d(nzb_soil:nzt_soil+1,1:surf_lsm%ns) )
!
!-- Allocate array to store maximum allowed timestep on each surface element. Later on,
!-- this will be used to limit the model time step accordingly. Note, in most of the cases
!-- the allowed soil timestep is much larger than the timestep allowed for the 3D flow model.
    ALLOCATE( surf_lsm%dt_max(1:surf_lsm%ns) )
!
!-- Allocate array for heat flux in W/m2, required for radiation?
!-- Consider to remove this array
    ALLOCATE( surf_lsm%surfhf(1:surf_lsm%ns) )
!
!-- Allocate intermediate timestep arrays
    ALLOCATE( tm_liq_m%var_1d(1:surf_lsm%ns) )
    ALLOCATE( tt_surface_m%var_1d(1:surf_lsm%ns) )
    ALLOCATE( tm_soil_m%var_2d(nzb_soil:nzt_soil,1:surf_lsm%ns) )
    ALLOCATE( tt_soil_m%var_2d(nzb_soil:nzt_soil,1:surf_lsm%ns) )
!
!-- Allocate 2D vegetation model arrays
    ALLOCATE( surf_lsm%building_surface(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%c_liq(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%c_surface(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%c_veg(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%f_sw_in(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%ghf(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%g_d(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%lai(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%lambda_surface_u(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%lambda_surface_s(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%nzt_pavement(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%pavement_surface(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%qsws_soil(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%qsws_liq(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%qsws_veg(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%rad_net_l(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%r_a(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%r_canopy(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%r_soil(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%r_soil_min(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%r_s(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%r_canopy_min(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%vegetation_surface(1:surf_lsm%ns) )
    ALLOCATE( surf_lsm%water_surface(1:surf_lsm%ns) )

    surf_lsm%water_surface      = .FALSE.
    surf_lsm%pavement_surface   = .FALSE.
    surf_lsm%vegetation_surface = .FALSE.
!
!-- Set default values
    surf_lsm%r_canopy_min = 0.0_wp
!
!-- Initial assignment of the pointers
!-- Horizontal surfaces
    t_soil    => t_soil_1;    t_soil_p    => t_soil_2
    t_surface => t_surface_1; t_surface_p => t_surface_2
    m_soil    => m_soil_1;    m_soil_p    => m_soil_2
    m_liq     => m_liq_1;     m_liq_p     => m_liq_2

 END SUBROUTINE lsm_init_arrays


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Parin for &lsmpar for land surface model
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_parin

    IMPLICIT NONE

    CHARACTER(LEN=100) ::  line  !< dummy string that contains the current line of the parameter
                                 !< file

    INTEGER(iwp)  ::  io_status  !< status after reading the nameslist file

    LOGICAL ::  switch_off_module = .FALSE.  !< local namelist parameter to switch off the module
                                             !< although the respective module namelist appears in
                                             !< the namelist file

    NAMELIST /land_surface_parameters/  aero_resist_kray,                                          &
                                        alpha_vangenuchten,                                        &
                                        c_surface,                                                 &
                                        canopy_resistance_coefficient,                             &
                                        constant_roughness,                                        &
                                        conserve_water_content,                                    &
                                        deep_soil_temperature,                                     &
                                        dz_soil,                                                   &
                                        f_shortwave_incoming,                                      &
                                        field_capacity,                                            &
                                        hydraulic_conductivity,                                    &
                                        lambda_surface_stable,                                     &
                                        lambda_surface_unstable,                                   &
                                        leaf_area_index,                                           &
                                        l_vangenuchten,                                            &
                                        min_canopy_resistance,                                     &
                                        min_soil_resistance,                                       &
                                        n_vangenuchten,                                            &
                                        pavement_depth_level,                                      &
                                        pavement_heat_capacity,                                    &
                                        pavement_heat_conduct,                                     &
                                        pavement_type,                                             &
                                        residual_moisture,                                         &
                                        root_fraction,                                             &
                                        saturation_moisture,                                       &
                                        skip_time_do_lsm,                                          &
                                        soil_moisture,                                             &
                                        soil_temperature,                                          &
                                        soil_type,                                                 &
                                        surface_type,                                              &
                                        switch_off_module,                                         &
                                        vegetation_coverage,                                       &
                                        vegetation_type,                                           &
                                        water_temperature,                                         &
                                        water_type,                                                &
                                        wilting_point,                                             &
                                        z0_pavement,                                               &
                                        z0_vegetation,                                             &
                                        z0_water,                                                  &
                                        z0h_pavement,                                              &
                                        z0h_vegetation,                                            &
                                        z0h_water,                                                 &
                                        z0q_pavement,                                              &
                                        z0q_vegetation,                                            &
                                        z0q_water

!
!-- Move to the beginning of the namelist file and try to find and read the namelist.
    REWIND( 11 )
    READ( 11, land_surface_parameters, IOSTAT=io_status )

!
!-- Action depending on the READ status
    IF ( io_status == 0 )  THEN
!
!--    land_surface_parameters namelist was found and read correctly. Set flag that indicates that
!--    the land surface model is switched on.
       IF ( .NOT. switch_off_module )  land_surface = .TRUE.

    ELSEIF ( io_status > 0 )  THEN
!
!--    land_surface_parameters namelist was found but contained errors. Print an error message
!--    including the line that caused the problem.
       BACKSPACE( 11 )
       READ( 11 , '(A)') line
       CALL parin_fail_message( 'land_surface_parameters', line )

    ENDIF

 END SUBROUTINE lsm_parin


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Soil model as part of the land surface model. The model predicts soil temperature and water
!> content.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_soil_model( calc_soil_moisture )


    IMPLICIT NONE

    INTEGER(iwp) ::  k       !< running index
    INTEGER(iwp) ::  m       !< running index

    LOGICAL, INTENT(IN) ::  calc_soil_moisture !< flag indicating whether soil moisture shall be calculated or not.

    REAL(wp) ::  h_vg !< Van Genuchten coef. h

    REAL(wp), DIMENSION(1:surf_lsm%ns) ::  m_total !< total soil water content

    REAL(wp), DIMENSION(1:surf_lsm%ns,nzb_soil:nzt_soil) ::  gamma_temp   !< temp. gamma
    REAL(wp), DIMENSION(1:surf_lsm%ns,nzb_soil:nzt_soil) ::  lambda_temp  !< temp. lambda
    REAL(wp), DIMENSION(1:surf_lsm%ns,nzb_soil:nzt_soil) ::  max_ts       !< maximum allowed timestep according diffusion criterion
    REAL(wp), DIMENSION(1:surf_lsm%ns,nzb_soil:nzt_soil) ::  tend         !< tendency


    TYPE(surf_type_lsm), POINTER ::  surf_m_soil
    TYPE(surf_type_lsm), POINTER ::  surf_m_soil_p
    TYPE(surf_type_lsm), POINTER ::  surf_t_soil
    TYPE(surf_type_lsm), POINTER ::  surf_t_soil_p
    TYPE(surf_type_lsm), POINTER ::  surf_tm_soil_m
    TYPE(surf_type_lsm), POINTER ::  surf_tt_soil_m

    TYPE(surf_type), POINTER  ::  surf  !< surface-date type variable


    IF ( debug_output_timestep )  THEN
       WRITE( debug_string, * ) 'lsm_soil_model', calc_soil_moisture
       CALL debug_message( debug_string, 'start' )
    ENDIF

    surf           => surf_lsm
    surf_m_soil    => m_soil
    surf_m_soil_p  => m_soil_p
    surf_t_soil    => t_soil
    surf_t_soil_p  => t_soil_p
    surf_tm_soil_m => tm_soil_m
    surf_tt_soil_m => tt_soil_m
!
!-- The diffusion criterion won't be evaluated at water surfaces. To guarantee that water surfaces
!-- do not impact the timestep, set the maximum allowed timestep to a huge number.
    IF ( intermediate_timestep_count == intermediate_timestep_count_max )                          &
       surf_lsm%dt_max = HUGE( 1.0_wp )
!
!-- In order to allow for vectorization on machines, the m loop is the inner loop, because the
!-- k loop has too few elements (insufficient vector length).
    !$OMP PARALLEL DO PRIVATE (m, k, lambda_temp, lambda_h_sat, ke) SCHEDULE (STATIC)
    DO  k = nzb_soil, nzt_soil

       DO  m = 1, surf%ns

          IF (  .NOT.  surf%water_surface(m) )  THEN
             IF ( surf%pavement_surface(m)  .AND.  k <= surf%nzt_pavement(m) )  THEN
                surf%rho_c_total(k,m) = surf%rho_c_total_def(k,m)
                lambda_temp(m,k) = surf%lambda_h_def(k,m)
             ELSE
!
!--             Calculate volumetric heat capacity of the soil, taking into account water content.
                surf%rho_c_total(k,m) = ( rho_c_soil * ( 1.0_wp - surf%m_sat(k,m) ) +              &
                                          rho_c_water * surf_m_soil%var_2d(k,m) )

!
!--             Calculate soil heat conductivity at the center of the soil layers.
                lambda_h_sat = lambda_h_sm**( 1.0_wp - surf%m_sat(k,m) ) *                         &
                               lambda_h_water**surf_m_soil%var_2d(k,m)

                ke = 1.0_wp + LOG10( MAX( 0.1_wp, surf_m_soil%var_2d(k,m) / surf%m_sat(k,m) ) )

                lambda_temp(m,k) = ke * ( lambda_h_sat - lambda_h_dry ) + lambda_h_dry
             ENDIF
          ENDIF
       ENDDO
    ENDDO

    IF ( surf%ns > 0 )  THEN
!
!--    Calculate soil heat conductivity (lambda_h) at the top layer level using the weighted
!--    average. For pavement surface, the true pavement depth is considered.
       !$OMP PARALLEL DO PRIVATE (k) SCHEDULE (STATIC)
       DO  k = nzb_soil, nzt_soil-1
          surf%lambda_h(k,1:surf%ns) = ( lambda_temp(1:surf%ns,k)   * dz_soil(k) +                 &
                                         lambda_temp(1:surf%ns,k+1) * dz_soil(k+1) ) *             &
                                       0.5_wp * ddz_soil_center(k)

       ENDDO
       surf%lambda_h(nzt_soil,1:surf%ns) = lambda_temp(1:surf%ns,nzt_soil)

!
!--    Prognostic equation for soil temperature t_soil
       tend = 0.0_wp

       WHERE( .NOT.  surf%water_surface(1:surf%ns) )
          tend(:,nzb_soil) = ( 1.0_wp / surf%rho_c_total(nzb_soil,1:surf%ns) ) *                   &
                             ( surf%lambda_h(nzb_soil,1:surf%ns) *                                 &
                               ( surf_t_soil%var_2d(nzb_soil+1,1:surf%ns) -                        &
                                 surf_t_soil%var_2d(nzb_soil,1:surf%ns) ) *                        &
                               ddz_soil_center(nzb_soil) + surf%ghf(1:surf%ns) ) *                 &
                               ddz_soil(nzb_soil)
       END WHERE

    ENDIF

    !$OMP PARALLEL DO PRIVATE (m, k) SCHEDULE (STATIC)
!NEC$ nointerchange
    DO  k = nzb_soil+1, nzt_soil
       DO  m = 1, surf%ns
          IF ( .NOT. surf%water_surface(m) )  THEN
             tend(m,k) = ( 1.0_wp / surf%rho_c_total(k,m) ) *                                      &
                         (   surf%lambda_h(k,m) *                                                  &
                             ( surf_t_soil%var_2d(k+1,m) - surf_t_soil%var_2d(k,m) ) *             &
                             ddz_soil_center(k) -                                                  &
                             surf%lambda_h(k-1,m) *                                                &
                             ( surf_t_soil%var_2d(k,m) - surf_t_soil%var_2d(k-1,m) ) *             &
                             ddz_soil_center(k-1)                                                  &
                         ) * ddz_soil(k)
          ENDIF
       ENDDO
    ENDDO

    !$OMP PARALLEL DO PRIVATE (m, k) SCHEDULE (STATIC)
!NEC$ nointerchange
    DO  k = nzb_soil, nzt_soil
       DO  m = 1, surf%ns
          IF ( .NOT. surf%water_surface(m) )  THEN
             surf_t_soil_p%var_2d(k,m) = surf_t_soil%var_2d(k,m) + dt_3d *                         &
                                         ( tsc(2) * tend(m,k) +                                    &
                                           tsc(3) * surf_tt_soil_m%var_2d(k,m) )
          ENDIF
       ENDDO
    ENDDO

!
!-- Calculate t_soil tendencies for the next Runge-Kutta step
    IF ( timestep_scheme(1:5) == 'runge' )  THEN
       IF ( intermediate_timestep_count == 1 )  THEN
          !$OMP PARALLEL DO PRIVATE (m, k) SCHEDULE (STATIC)
!NEC$ nointerchange
          DO  k = nzb_soil, nzt_soil
             DO  m = 1, surf%ns
                IF ( .NOT. surf%water_surface(m) )  THEN
                   surf_tt_soil_m%var_2d(k,m) = tend(m,k)
                ENDIF
             ENDDO
          ENDDO
       ELSEIF ( intermediate_timestep_count < intermediate_timestep_count_max )  THEN
!NEC$ nointerchange
          DO  k = nzb_soil, nzt_soil
             DO  m = 1, surf%ns
                IF ( .NOT. surf%water_surface(m) )  THEN
                   surf_tt_soil_m%var_2d(k,m) = -9.5625_wp * tend(m,k) +                           &
                                                 5.3125_wp * surf_tt_soil_m%var_2d(k,m)
                ENDIF
             ENDDO
          ENDDO
       ENDIF
    ENDIF

    IF ( calc_soil_moisture )  THEN

       !$OMP PARALLEL DO PRIVATE (m, k, h_vg ) SCHEDULE (STATIC)
 !NEC$ nointerchange
       DO  k = nzb_soil, nzt_soil
          DO  m = 1, surf%ns
             IF ( .NOT. surf%water_surface(m) )  THEN
 !
 !--            In order to prevent water tranport through paved surfaces, conductivity and
 !--            diffusivity are set to zero.
                IF ( surf%pavement_surface(m)  .AND.  k <= surf%nzt_pavement(m) )  THEN
                   lambda_temp(m,k) = 0.0_wp
                   gamma_temp(m,k)  = 0.0_wp
                ELSE
!
!--                Calculate soil diffusivity at the center of the soil layers.
                   lambda_temp(m,k) = ( -b_ch * surf%gamma_w_sat(k,m) *                            &
                                        psi_sat / surf%m_sat(k,m) ) *                              &
                                      ( MAX( surf_m_soil%var_2d(k,m), surf%m_wilt(k,m) ) /         &
                                        surf%m_sat(k,m) )**( b_ch + 2.0_wp )

!
!--                Parametrization of Van Genuchten.
!--                Calculate the hydraulic conductivity after Van Genuchten (1980).
                   h_vg = ( ( ( surf%m_res(k,m) - surf%m_sat(k,m) ) /                              &
                              ( surf%m_res(k,m) - MAX( surf_m_soil%var_2d(k,m), surf%m_wilt(k,m) ) &
                              )                                                                    &
                            )**( surf%n_vg(k,m) / ( surf%n_vg(k,m) - 1.0_wp ) ) - 1.0_wp           &
                          )**( 1.0_wp / surf%n_vg(k,m) ) / surf%alpha_vg(k,m)

                   gamma_temp(m,k) = surf%gamma_w_sat(k,m) * ( (                                   &
                                     ( 1.0_wp + ( surf%alpha_vg(k,m) * h_vg )**surf%n_vg(k,m)      &
                                     )**( 1.0_wp - 1.0_wp / surf%n_vg(k,m) ) -                     &
                                     ( surf%alpha_vg(k,m) * h_vg )**( surf%n_vg(k,m) - 1.0_wp )    &
                                                               )**2                                &
                                                             ) /                                   &
                                                               (                                   &
                                     ( 1.0_wp + ( surf%alpha_vg(k,m) * h_vg )**surf%n_vg(k,m)      &
                                     )**( ( 1.0_wp  - 1.0_wp / surf%n_vg(k,m) ) *                  &
                                     ( surf%l_vg(k,m) + 2.0_wp ) )                                 &
                                                               )

                ENDIF
             ENDIF
          ENDDO
       ENDDO
    ENDIF

    IF ( calc_soil_moisture )  THEN
!
!--    Prognostic equation for soil moisture content. Only performed, when humidity is enabled
!--    in the atmosphere.
       IF ( humidity )  THEN
!
!--       Calculate soil diffusivity (lambda_w) at the _layer level using linear
!--       interpolation. To do: replace this with ECMWF-IFS Eq. 8.81.
          !$OMP PARALLEL DO PRIVATE (m, k) SCHEDULE (STATIC)
!NEC$ nointerchange
          DO  k = nzb_soil, nzt_soil-1
             DO  m = 1, surf%ns
                IF ( .NOT. surf%water_surface(m) )  THEN
                   surf%lambda_w(k,m) = ( lambda_temp(m,k)   * dz_soil(k) +                        &
                                          lambda_temp(m,k+1) * dz_soil(k+1) ) *                    &
                                        0.5_wp * ddz_soil_center(k)
                   surf%gamma_w(k,m) = ( gamma_temp(m,k)   * dz_soil(k) +                          &
                                         gamma_temp(m,k+1) * dz_soil(k+1) ) *                      &
                                       0.5_wp * ddz_soil_center(k)
                ENDIF
             ENDDO
          ENDDO
!
!--       In case of a closed bottom (= water content is conserved), set hydraulic
!--       conductivity to zero so that no water will be lost in the bottom layer. As gamma_w
!--       is always a positive value, it cannot be set to zero in case of purely dry soil
!--       since this would cause accumulation of (non-existing) water in the lowest soil layer.
          !$OMP PARALLEL DO PRIVATE (m) SCHEDULE (STATIC)
          DO  m = 1, surf%ns
             IF ( .NOT. surf%water_surface(m) )  THEN
                IF ( conserve_water_content  .AND.  surf_m_soil%var_2d(nzt_soil,m) /= 0.0_wp )  THEN
                   surf%gamma_w(nzt_soil,m) = 0.0_wp
                ELSE
                   surf%gamma_w(nzt_soil,m) = gamma_temp(m,nzt_soil)
                ENDIF
             ENDIF
          ENDDO

          m_total = 0.0_wp
          !$OMP PARALLEL DO PRIVATE (m, k) SCHEDULE (STATIC)
!NEC$ nointerchange
          DO  k = nzb_soil, nzt_soil
             DO  m = 1, surf%ns
                IF ( .NOT. surf%water_surface(m) )  THEN
                   IF ( surf_m_soil%var_2d(k,m) > surf%m_wilt(k,m) )  THEN
                      m_total(m) = m_total(m) + surf%root_fr(k,m) * surf_m_soil%var_2d(k,m)
                   ENDIF
                ENDIF
             ENDDO
          ENDDO
!
!--       The root extraction (= root_extr * qsws_veg / (rho_l * l_v)) ensures the mass
!--       conservation for water. The transpiration of plants equals the cumulative
!--       withdrawals by the roots in the soil. The scheme takes into account the
!--       availability of water in the soil layers as well as the root fraction in the
!--       respective layer. Layer with moisture below wilting point will not contribute, which
!--       reflects the preference of plants to take water from moister layers.
!--       Calculate the root extraction (ECMWF 7.69, the sum of root_extr = 1). The energy
!--       balance solver guarantees a positive transpiration, so that there is no need for an
!--       additional check.
          !$OMP PARALLEL DO PRIVATE (m, k) SCHEDULE (STATIC)
!NEC$ nointerchange
          DO  k = nzb_soil, nzt_soil
             DO  m = 1, surf%ns
                IF ( .NOT. surf%water_surface(m) )  THEN
                   IF ( m_total(m) > 0.0_wp )  THEN
                      IF ( surf_m_soil%var_2d(k,m) > surf%m_wilt(k,m) )  THEN
                         root_extr(k) = surf%root_fr(k,m) * surf_m_soil%var_2d(k,m) / m_total(m)
                      ELSE
                         root_extr(k) = 0.0_wp
                      ENDIF
                   ENDIF
                ENDIF
             ENDDO
          ENDDO
!
!--       Prognostic term for soil water content m_soil.
          tend = 0.0_wp
          !$OMP PARALLEL DO PRIVATE (m, k) SCHEDULE (STATIC)
          DO  m = 1, surf%ns
             IF ( .NOT. surf%water_surface(m) )  THEN
                tend(m,nzb_soil) = ( surf%lambda_w(nzb_soil,m) *                                   &
                                     ( surf_m_soil%var_2d(nzb_soil+1,m) -                          &
                                       surf_m_soil%var_2d(nzb_soil,m)                              &
                                     ) * ddz_soil_center(nzb_soil) -                               &
                                     surf%gamma_w(nzb_soil,m) -                                    &
                                     ( root_extr(nzb_soil) * surf%qsws_veg(m) +                    &
                                       surf%qsws_soil(m) ) * drho_l_lv                             &
                                   ) * ddz_soil(nzb_soil)
             ENDIF
          ENDDO

          !$OMP PARALLEL DO PRIVATE (m, k) SCHEDULE (STATIC)
!NEC$ nointerchange
          DO  k = nzb_soil+1, nzt_soil-1
             DO  m = 1, surf%ns
                IF ( .NOT. surf%water_surface(m) )  THEN
                   tend(m,k) = ( surf%lambda_w(k,m)   * ( surf_m_soil%var_2d(k+1,m) -              &
                                                          surf_m_soil%var_2d(k,m) )                &
                                                      * ddz_soil_center(k) - surf%gamma_w(k,m) -   &
                                 surf%lambda_w(k-1,m) * ( surf_m_soil%var_2d(k,m) -                &
                                                          surf_m_soil%var_2d(k-1,m) )              &
                                                      * ddz_soil_center(k-1) + surf%gamma_w(k-1,m) &
                               - ( root_extr(k) * surf%qsws_veg(m) * drho_l_lv )                   &
                               ) * ddz_soil(k)
                ENDIF
             ENDDO
          ENDDO
          !$OMP PARALLEL DO PRIVATE (m) SCHEDULE (STATIC)
          DO  m = 1, surf%ns
             IF ( .NOT. surf%water_surface(m) )  THEN
                tend(m,nzt_soil) = ( -surf%gamma_w(nzt_soil,m) - surf%lambda_w(nzt_soil-1,m) *    &
                                      ( surf_m_soil%var_2d(nzt_soil,m) -                          &
                                        surf_m_soil%var_2d(nzt_soil-1,m) ) *                      &
                                      ddz_soil_center(nzt_soil-1) + surf%gamma_w(nzt_soil-1,m) -  &
                                      ( root_extr(nzt_soil) * surf%qsws_veg(m) * drho_l_lv )      &
                                   ) * ddz_soil(nzt_soil)

             ENDIF
          ENDDO
!
!--       Actual time integration.
          !$OMP PARALLEL DO PRIVATE (m, k) SCHEDULE (STATIC)
!NEC$ nointerchange
          DO  k = nzb_soil, nzt_soil
             DO  m = 1, surf%ns
                IF ( .NOT. surf%water_surface(m) )  THEN
                   surf_m_soil_p%var_2d(k,m) = surf_m_soil%var_2d(k,m) + dt_3d *                   &
                                              ( tsc(2) * tend(m,k) +                               &
                                                tsc(3) * surf_tm_soil_m%var_2d(k,m) )
                ENDIF
             ENDDO
          ENDDO
!
!--       Account for dry and wet soils to keep solution stable (mass conservation is violated
!--       here).
          !$OMP PARALLEL DO PRIVATE (m, k) SCHEDULE (STATIC)
!NEC$ nointerchange
          DO  k = nzb_soil, nzt_soil
             DO  m = 1, surf%ns
                IF ( .NOT. surf%water_surface(m) )  THEN
                   surf_m_soil_p%var_2d(k,m) = MIN( surf_m_soil_p%var_2d(k,m), surf%m_sat(k,m) )
                   surf_m_soil_p%var_2d(k,m) = MAX( surf_m_soil_p%var_2d(k,m), 0.0_wp )
                ENDIF
             ENDDO
          ENDDO
!
!--       Calculate m_soil tendencies for the next Runge-Kutta step.
          IF ( timestep_scheme(1:5) == 'runge' )  THEN
             IF ( intermediate_timestep_count == 1 )  THEN
                !$OMP PARALLEL DO PRIVATE (m, k) SCHEDULE (STATIC)
!NEC$ nointerchange
                DO  k = nzb_soil, nzt_soil
                   DO  m = 1, surf%ns
                      IF ( .NOT. surf%water_surface(m) )  THEN
                         surf_tm_soil_m%var_2d(k,m) = tend(m,k)
                      ENDIF
                   ENDDO
                ENDDO
             ELSEIF ( intermediate_timestep_count < intermediate_timestep_count_max )  THEN
                !$OMP PARALLEL DO PRIVATE (m, k) SCHEDULE (STATIC)
 !NEC$ nointerchange
                DO  k = nzb_soil, nzt_soil
                   DO  m = 1, surf%ns
                      IF (  .NOT.  surf%water_surface(m) )  THEN
                         surf_tm_soil_m%var_2d(k,m) = -9.5625_wp * tend(m,k) +                     &
                                                       5.3125_wp * surf_tm_soil_m%var_2d(k,m)
                      ENDIF
                   ENDDO
                ENDDO
             ENDIF
          ENDIF
       ENDIF ! humidity
    ENDIF ! calc_soil_moisture
!
!-- Determine allowed timestep according a diffusion criterion. This is only required for
!-- the last substep.
    IF ( intermediate_timestep_count == intermediate_timestep_count_max )  THEN
       !$OMP PARALLEL DO PRIVATE (m, k) SCHEDULE (STATIC)
!NEC$ nointerchange
       DO  k = nzb_soil, nzt_soil
          DO  m = 1, surf%ns
             IF ( .NOT. surf%water_surface(m) )  THEN
                max_ts(m,k) = surf%rho_c_total(k,m) / surf%lambda_h(k,m) * ( dz_soil(k) )**2
             ENDIF
          ENDDO
       ENDDO
       DO  m = 1, surf%ns
          IF ( .NOT. surf%water_surface(m) )  THEN
             surf_lsm%dt_max(m) = MINVAL( max_ts(m,:) )
          ENDIF
       ENDDO
    ENDIF

!
!-- Debug location message
    IF ( debug_output_timestep )  THEN
       WRITE( debug_string, * )  'lsm_soil_model', calc_soil_moisture
       CALL debug_message( debug_string, 'end' )
    ENDIF

 END SUBROUTINE lsm_soil_model


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Swapping of timelevels
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_swap_timelevel ( mod_count )

    IMPLICIT NONE

    INTEGER, INTENT(IN) :: mod_count


    SELECT CASE ( mod_count )

       CASE ( 0 )
          t_surface  => t_surface_1; t_surface_p  => t_surface_2
          t_soil     => t_soil_1;    t_soil_p     => t_soil_2

          IF ( humidity )  THEN
             m_soil  => m_soil_1;    m_soil_p     => m_soil_2
             m_liq   => m_liq_1;     m_liq_p      => m_liq_2
          ENDIF

       CASE ( 1 )
          t_surface  => t_surface_2; t_surface_p  => t_surface_1
          t_soil     => t_soil_2;    t_soil_p     => t_soil_1

          IF ( humidity )  THEN
             m_soil  => m_soil_2;    m_soil_p     => m_soil_1
             m_liq   => m_liq_2;     m_liq_p      => m_liq_1

          ENDIF

    END SELECT

 END SUBROUTINE lsm_swap_timelevel


!--------------------------------------------------------------------------------------------------!
!
! Description:
! ------------
!> Subroutine for averaging 3D data
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_3d_data_averaging( mode, variable )


    USE control_parameters

    USE indices

    IMPLICIT NONE

    CHARACTER (LEN=*) ::  mode    !<
    CHARACTER (LEN=*) ::  variable !<

    INTEGER(iwp) ::  i       !<
    INTEGER(iwp) ::  j       !<
    INTEGER(iwp) ::  k       !<
    INTEGER(iwp) ::  m       !< running index

    IF ( mode == 'allocate' )  THEN

       SELECT CASE ( TRIM( variable ) )

             CASE ( 'c_liq*' )
                IF ( .NOT. ALLOCATED( c_liq_av ) )  THEN
                   ALLOCATE( c_liq_av(nysg:nyng,nxlg:nxrg) )
                ENDIF
                c_liq_av = 0.0_wp

             CASE ( 'c_soil*' )
                IF ( .NOT. ALLOCATED( c_soil_av ) )  THEN
                   ALLOCATE( c_soil_av(nysg:nyng,nxlg:nxrg) )
                ENDIF
                c_soil_av = 0.0_wp

             CASE ( 'c_veg*' )
                IF ( .NOT. ALLOCATED( c_veg_av ) )  THEN
                   ALLOCATE( c_veg_av(nysg:nyng,nxlg:nxrg) )
                ENDIF
                c_veg_av = 0.0_wp

             CASE ( 'lai*' )
                IF ( .NOT. ALLOCATED( lai_av ) )  THEN
                   ALLOCATE( lai_av(nysg:nyng,nxlg:nxrg) )
                ENDIF
                lai_av = 0.0_wp

             CASE ( 'm_liq*' )
                IF ( .NOT. ALLOCATED( m_liq_av ) )  THEN
                   ALLOCATE( m_liq_av(nysg:nyng,nxlg:nxrg) )
                ENDIF
                m_liq_av = 0.0_wp

             CASE ( 'm_soil' )
                IF ( .NOT. ALLOCATED( m_soil_av ) )  THEN
                   ALLOCATE( m_soil_av(nzb_soil:nzt_soil,nysg:nyng,nxlg:nxrg) )
                ENDIF
                m_soil_av = 0.0_wp

             CASE ( 'qsws_liq*' )
                IF ( .NOT. ALLOCATED( qsws_liq_av ) )  THEN
                   ALLOCATE( qsws_liq_av(nysg:nyng,nxlg:nxrg) )
                ENDIF
                qsws_liq_av = 0.0_wp

             CASE ( 'qsws_soil*' )
                IF ( .NOT. ALLOCATED( qsws_soil_av ) )  THEN
                   ALLOCATE( qsws_soil_av(nysg:nyng,nxlg:nxrg) )
                ENDIF
                qsws_soil_av = 0.0_wp

             CASE ( 'qsws_veg*' )
                IF ( .NOT. ALLOCATED( qsws_veg_av ) )  THEN
                   ALLOCATE( qsws_veg_av(nysg:nyng,nxlg:nxrg) )
                ENDIF
                qsws_veg_av = 0.0_wp

             CASE ( 'r_s*' )
                IF ( .NOT. ALLOCATED( r_s_av ) )  THEN
                   ALLOCATE( r_s_av(nysg:nyng,nxlg:nxrg) )
                ENDIF
                r_s_av = 0.0_wp

             CASE ( 't_soil' )
                IF ( .NOT. ALLOCATED( t_soil_av ) )  THEN
                   ALLOCATE( t_soil_av(nzb_soil:nzt_soil,nysg:nyng,nxlg:nxrg) )
                ENDIF
                t_soil_av = 0.0_wp

          CASE DEFAULT
             CONTINUE

       END SELECT

    ELSEIF ( mode == 'sum' )  THEN
!
!--    Note, as horizontal cross-section output is only 2D, only sum-up data from
!--    horizontally upward-facing surfaces.
       SELECT CASE ( TRIM( variable ) )

          CASE ( 'c_liq*' )
             IF ( ALLOCATED( c_liq_av ) )  THEN
                DO  m = 1, surf_lsm%ns
                   i   = surf_lsm%i(m)
                   j   = surf_lsm%j(m)
                   c_liq_av(j,i) = c_liq_av(j,i) +                                                 &
                                            MERGE( surf_lsm%c_liq(m), 0.0_wp, surf_lsm%upward(m) )
                ENDDO
             ENDIF

          CASE ( 'c_soil*' )
             IF ( ALLOCATED( c_soil_av ) )  THEN
                DO  m = 1, surf_lsm%ns
                   i   = surf_lsm%i(m)
                   j   = surf_lsm%j(m)
                   c_soil_av(j,i) = c_soil_av(j,i) +                                               &
                                    MERGE( (1.0 - surf_lsm%c_veg(m)), 0.0_wp, surf_lsm%upward(m) )
                ENDDO
             ENDIF

          CASE ( 'c_veg*' )
             IF ( ALLOCATED( c_veg_av ) )  THEN
                DO  m = 1, surf_lsm%ns
                   i   = surf_lsm%i(m)
                   j   = surf_lsm%j(m)
                   c_veg_av(j,i) = c_veg_av(j,i) +                                                 &
                                            MERGE( surf_lsm%c_veg(m), 0.0_wp, surf_lsm%upward(m) )
                ENDDO
             ENDIF

          CASE ( 'lai*' )
             IF ( ALLOCATED( lai_av ) )  THEN
                DO  m = 1, surf_lsm%ns
                   i   = surf_lsm%i(m)
                   j   = surf_lsm%j(m)
                   lai_av(j,i) = lai_av(j,i) +                                                     &
                                              MERGE( surf_lsm%lai(m), 0.0_wp, surf_lsm%upward(m) )
                ENDDO
             ENDIF

          CASE ( 'm_liq*' )
             IF ( ALLOCATED( m_liq_av ) )  THEN
                DO  m = 1, surf_lsm%ns
                   i   = surf_lsm%i(m)
                   j   = surf_lsm%j(m)
                   m_liq_av(j,i) = m_liq_av(j,i) +                                                 &
                                              MERGE( m_liq%var_1d(m), 0.0_wp, surf_lsm%upward(m) )
                ENDDO
             ENDIF

          CASE ( 'm_soil' )
             IF ( ALLOCATED( m_soil_av ) )  THEN
                DO  m = 1, surf_lsm%ns
                   i   = surf_lsm%i(m)
                   j   = surf_lsm%j(m)
                   DO  k = nzb_soil, nzt_soil
                      m_soil_av(k,j,i) = m_soil_av(k,j,i) +                                        &
                                           MERGE( m_soil%var_2d(k,m), 0.0_wp, surf_lsm%upward(m) )
                   ENDDO
                ENDDO
             ENDIF

          CASE ( 'qsws_liq*' )
             IF ( ALLOCATED( qsws_liq_av ) )  THEN
                DO  m = 1, surf_lsm%ns
                   i   = surf_lsm%i(m)
                   j   = surf_lsm%j(m)
                   qsws_liq_av(j,i) = qsws_liq_av(j,i) +                                           &
                                         MERGE( surf_lsm%qsws_liq(m), 0.0_wp, surf_lsm%upward(m) )
                ENDDO
             ENDIF

          CASE ( 'qsws_soil*' )
             IF ( ALLOCATED( qsws_soil_av ) )  THEN
                DO  m = 1, surf_lsm%ns
                   i   = surf_lsm%i(m)
                   j   = surf_lsm%j(m)
                   qsws_soil_av(j,i) = qsws_soil_av(j,i) +                                         &
                                        MERGE( surf_lsm%qsws_soil(m), 0.0_wp, surf_lsm%upward(m) )
                ENDDO
             ENDIF

          CASE ( 'qsws_veg*' )
             IF ( ALLOCATED(qsws_veg_av ) )  THEN
                DO  m = 1, surf_lsm%ns
                   i   = surf_lsm%i(m)
                   j   = surf_lsm%j(m)
                   qsws_veg_av(j,i) = qsws_veg_av(j,i) +                                           &
                                         MERGE( surf_lsm%qsws_veg(m), 0.0_wp, surf_lsm%upward(m) )
                ENDDO
             ENDIF

          CASE ( 'r_s*' )
             IF ( ALLOCATED( r_s_av) )  THEN
                DO  m = 1, surf_lsm%ns
                   i   = surf_lsm%i(m)
                   j   = surf_lsm%j(m)
                   r_s_av(j,i) = r_s_av(j,i) +                                                     &
                                              MERGE( surf_lsm%r_s(m), 0.0_wp, surf_lsm%upward(m) )
                ENDDO
             ENDIF

          CASE ( 't_soil' )
             IF ( ALLOCATED( t_soil_av ) )  THEN
                DO  m = 1, surf_lsm%ns
                   i   = surf_lsm%i(m)
                   j   = surf_lsm%j(m)
                   DO  k = nzb_soil, nzt_soil
                      t_soil_av(k,j,i) = t_soil_av(k,j,i) +                                        &
                                           MERGE( t_soil%var_2d(k,m), 0.0_wp, surf_lsm%upward(m) )
                   ENDDO
                ENDDO
             ENDIF

          CASE DEFAULT
             CONTINUE

       END SELECT

    ELSEIF ( mode == 'average' )  THEN

       SELECT CASE ( TRIM( variable ) )

          CASE ( 'c_liq*' )
             IF ( ALLOCATED( c_liq_av ) )  THEN
                DO  i = nxl, nxr
                   DO  j = nys, nyn
                      c_liq_av(j,i) = c_liq_av(j,i) / REAL( average_count_3d, KIND=wp )
                   ENDDO
                ENDDO
             ENDIF

          CASE ( 'c_soil*' )
             IF ( ALLOCATED( c_soil_av ) )  THEN
                DO  i = nxl, nxr
                   DO  j = nys, nyn
                      c_soil_av(j,i) = c_soil_av(j,i) / REAL( average_count_3d, KIND=wp )
                   ENDDO
                ENDDO
             ENDIF

          CASE ( 'c_veg*' )
             IF ( ALLOCATED( c_veg_av ) )  THEN
                DO  i = nxl, nxr
                   DO  j = nys, nyn
                      c_veg_av(j,i) = c_veg_av(j,i) / REAL( average_count_3d, KIND=wp )
                   ENDDO
                ENDDO
             ENDIF

         CASE ( 'lai*' )
             IF ( ALLOCATED( lai_av ) )  THEN
                DO  i = nxl, nxr
                   DO  j = nys, nyn
                      lai_av(j,i) = lai_av(j,i) / REAL( average_count_3d, KIND=wp )
                   ENDDO
                ENDDO
             ENDIF

          CASE ( 'm_liq*' )
             IF ( ALLOCATED( m_liq_av ) )  THEN
                DO  i = nxl, nxr
                   DO  j = nys, nyn
                      m_liq_av(j,i) = m_liq_av(j,i) / REAL( average_count_3d, KIND=wp )
                   ENDDO
                ENDDO
             ENDIF

          CASE ( 'm_soil' )
             IF ( ALLOCATED( m_soil_av ) )  THEN
                DO  i = nxl, nxr
                   DO  j = nys, nyn
                      DO  k = nzb_soil, nzt_soil
                         m_soil_av(k,j,i) = m_soil_av(k,j,i) / REAL( average_count_3d, KIND=wp )
                      ENDDO
                   ENDDO
                ENDDO
             ENDIF

          CASE ( 'qsws_liq*' )
             IF ( ALLOCATED( qsws_liq_av ) )  THEN
                DO  i = nxl, nxr
                   DO  j = nys, nyn
                      qsws_liq_av(j,i) = qsws_liq_av(j,i) / REAL( average_count_3d, KIND=wp )
                   ENDDO
                ENDDO
             ENDIF

          CASE ( 'qsws_soil*' )
             IF ( ALLOCATED( qsws_soil_av ) )  THEN
                DO  i = nxl, nxr
                   DO  j = nys, nyn
                      qsws_soil_av(j,i) = qsws_soil_av(j,i) / REAL( average_count_3d, KIND=wp )
                   ENDDO
                ENDDO
             ENDIF

          CASE ( 'qsws_veg*' )
             IF ( ALLOCATED( qsws_veg_av ) )  THEN
                DO  i = nxl, nxr
                   DO  j = nys, nyn
                      qsws_veg_av(j,i) = qsws_veg_av(j,i) / REAL( average_count_3d, KIND=wp )
                   ENDDO
                ENDDO
             ENDIF

          CASE ( 'r_s*' )
             IF ( ALLOCATED( r_s_av ) )  THEN
                DO  i = nxl, nxr
                   DO  j = nys, nyn
                      r_s_av(j,i) = r_s_av(j,i) / REAL( average_count_3d, KIND=wp )
                   ENDDO
                ENDDO
             ENDIF

          CASE ( 't_soil' )
             IF ( ALLOCATED( t_soil_av ) )  THEN
                DO  i = nxl, nxr
                   DO  j = nys, nyn
                      DO  k = nzb_soil, nzt_soil
                         t_soil_av(k,j,i) = t_soil_av(k,j,i) / REAL( average_count_3d, KIND=wp )
                      ENDDO
                   ENDDO
                ENDDO
             ENDIF
!
!--

       END SELECT

    ENDIF

 END SUBROUTINE lsm_3d_data_averaging


!--------------------------------------------------------------------------------------------------!
!
! Description:
! ------------
!> Subroutine defining appropriate grid for netcdf variables.
!> It is called out from subroutine netcdf.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_define_netcdf_grid( var, found, grid_x, grid_y, grid_z )

     IMPLICIT NONE

     CHARACTER (LEN=*), INTENT(IN)  ::  var         !<

     CHARACTER (LEN=*), INTENT(OUT) ::  grid_x      !<
     CHARACTER (LEN=*), INTENT(OUT) ::  grid_y      !<
     CHARACTER (LEN=*), INTENT(OUT) ::  grid_z      !<

     LOGICAL, INTENT(OUT)           ::  found       !<

     found  = .TRUE.

!
!--  Check for the grid
     SELECT CASE ( TRIM( var ) )

        CASE ( 'm_soil', 't_soil', 'm_soil_xy', 't_soil_xy', 'm_soil_xz', 't_soil_xz', 'm_soil_yz',&
               't_soil_yz' )
           grid_x = 'x'
           grid_y = 'y'
           grid_z = 'zs'

        CASE DEFAULT
           found  = .FALSE.
           grid_x = 'none'
           grid_y = 'none'
           grid_z = 'none'
     END SELECT

 END SUBROUTINE lsm_define_netcdf_grid

!--------------------------------------------------------------------------------------------------!
!
! Description:
! ------------
!> Subroutine defining 3D output variables
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_data_output_2d( av, variable, found, grid, mode, local_pf, two_d, nzb_do, nzt_do )

    USE indices


    IMPLICIT NONE

    CHARACTER (LEN=*) ::  grid     !<
    CHARACTER (LEN=*) ::  mode     !<
    CHARACTER (LEN=*) ::  variable !<

    INTEGER(iwp) ::  av      !<
    INTEGER(iwp) ::  i       !< running index
    INTEGER(iwp) ::  j       !< running index
    INTEGER(iwp) ::  k       !< running index
    INTEGER(iwp) ::  m       !< running index
    INTEGER(iwp) ::  nzb_do  !<
    INTEGER(iwp) ::  nzt_do  !<

    LOGICAL      ::  found !<
    LOGICAL      ::  two_d !< flag parameter that indicates 2D variables (horizontal cross sections)

    REAL(wp), DIMENSION(nxl:nxr,nys:nyn,nzb_do:nzt_do) ::  local_pf !<


    found = .TRUE.

    SELECT CASE ( TRIM( variable ) )
!
!--    Before data is transfered to local_pf, transfer is it 2D dummy variable and exchange ghost
!--    points therein.
!--    However, at this point this is only required for instantaneous arrays, time-averaged
!--    quantities are already exchanged.
       CASE ( 'c_liq*_xy' )        ! 2d-array
          IF ( av == 0 )  THEN
             DO  m = 1, surf_lsm%ns
                i                   = surf_lsm%i(m)
                j                   = surf_lsm%j(m)
                local_pf(i,j,nzb+1) = MERGE( surf_lsm%c_liq(m), local_pf(i,j,nzb+1),               &
                                             surf_lsm%upward(m) )
             ENDDO
          ELSE
            IF ( .NOT. ALLOCATED( c_liq_av ) )  THEN
               ALLOCATE( c_liq_av(nysg:nyng,nxlg:nxrg) )
               c_liq_av = 0.0_wp
            ENDIF
             DO  i = nxl, nxr
                DO  j = nys, nyn
                   local_pf(i,j,nzb+1) = c_liq_av(j,i)
                ENDDO
             ENDDO
          ENDIF

          two_d = .TRUE.
          grid = 'zu1'

       CASE ( 'c_soil*_xy' )        ! 2d-array
          IF ( av == 0 )  THEN
             DO  m = 1, surf_lsm%ns
                i                   = surf_lsm%i(m)
                j                   = surf_lsm%j(m)
                local_pf(i,j,nzb+1) = MERGE( 1.0_wp - surf_lsm%c_veg(m), local_pf(i,j,nzb+1),      &
                                             surf_lsm%upward(m) )
             ENDDO
          ELSE
            IF ( .NOT. ALLOCATED( c_soil_av ) )  THEN
               ALLOCATE( c_soil_av(nysg:nyng,nxlg:nxrg) )
               c_soil_av = 0.0_wp
            ENDIF
             DO  i = nxl, nxr
                DO  j = nys, nyn
                   local_pf(i,j,nzb+1) = c_soil_av(j,i)
                ENDDO
             ENDDO
          ENDIF

          two_d = .TRUE.
          grid = 'zu1'

       CASE ( 'c_veg*_xy' )        ! 2d-array
          IF ( av == 0 )  THEN
             DO  m = 1, surf_lsm%ns
                i                   = surf_lsm%i(m)
                j                   = surf_lsm%j(m)
                local_pf(i,j,nzb+1) = MERGE( surf_lsm%c_veg(m), local_pf(i,j,nzb+1),               &
                                             surf_lsm%upward(m) )
             ENDDO
          ELSE
            IF ( .NOT. ALLOCATED( c_veg_av ) )  THEN
               ALLOCATE( c_veg_av(nysg:nyng,nxlg:nxrg) )
               c_veg_av = 0.0_wp
            ENDIF
             DO  i = nxl, nxr
                DO  j = nys, nyn
                   local_pf(i,j,nzb+1) = c_veg_av(j,i)
                ENDDO
             ENDDO
          ENDIF

          two_d = .TRUE.
          grid = 'zu1'

       CASE ( 'lai*_xy' )        ! 2d-array
          IF ( av == 0 )  THEN
             DO  m = 1, surf_lsm%ns
                i                   = surf_lsm%i(m)
                j                   = surf_lsm%j(m)
                local_pf(i,j,nzb+1) = MERGE( surf_lsm%lai(m), local_pf(i,j,nzb+1),                 &
                                             surf_lsm%upward(m) )
             ENDDO
          ELSE
            IF ( .NOT. ALLOCATED( lai_av ) )  THEN
               ALLOCATE( lai_av(nysg:nyng,nxlg:nxrg) )
               lai_av = 0.0_wp
            ENDIF
             DO  i = nxl, nxr
                DO  j = nys, nyn
                   local_pf(i,j,nzb+1) = lai_av(j,i)
                ENDDO
             ENDDO
          ENDIF

          two_d = .TRUE.
          grid = 'zu1'

       CASE ( 'm_liq*_xy' )        ! 2d-array
          IF ( av == 0 )  THEN
             DO  m = 1, surf_lsm%ns
                i                   = surf_lsm%i(m)
                j                   = surf_lsm%j(m)
                local_pf(i,j,nzb+1) = MERGE( m_liq%var_1d(m), local_pf(i,j,nzb+1),                 &
                                             surf_lsm%upward(m) )
             ENDDO
          ELSE
            IF ( .NOT. ALLOCATED( m_liq_av ) )  THEN
               ALLOCATE( m_liq_av(nysg:nyng,nxlg:nxrg) )
               m_liq_av = 0.0_wp
            ENDIF
             DO  i = nxl, nxr
                DO  j = nys, nyn
                   local_pf(i,j,nzb+1) = m_liq_av(j,i)
                ENDDO
             ENDDO
          ENDIF

          two_d = .TRUE.
          grid = 'zu1'

       CASE ( 'm_soil_xy', 'm_soil_xz', 'm_soil_yz' )
          IF ( av == 0 )  THEN
             DO  m = 1, surf_lsm%ns
                i   = surf_lsm%i(m)
                j   = surf_lsm%j(m)
                DO k = nzb_soil, nzt_soil
                   local_pf(i,j,k) = MERGE( m_soil%var_2d(k,m), local_pf(i,j,k),                   &
                                            surf_lsm%upward(m) )
                ENDDO
             ENDDO
          ELSE
            IF ( .NOT. ALLOCATED( m_soil_av ) )  THEN
               ALLOCATE( m_soil_av(nzb_soil:nzt_soil,nysg:nyng,nxlg:nxrg) )
               m_soil_av = 0.0_wp
            ENDIF
             DO  i = nxl, nxr
                DO  j = nys, nyn
                   DO k = nzb_soil, nzt_soil
                      local_pf(i,j,k) = m_soil_av(k,j,i)
                   ENDDO
                ENDDO
             ENDDO
          ENDIF

          nzb_do = nzb_soil
          nzt_do = nzt_soil

          IF ( mode == 'xy' ) grid = 'zs'

       CASE ( 'qsws_liq*_xy' )        ! 2d-array
          IF ( av == 0 )  THEN
             DO  m = 1, surf_lsm%ns
                i                   = surf_lsm%i(m)
                j                   = surf_lsm%j(m)
                local_pf(i,j,nzb+1) = MERGE( surf_lsm%qsws_liq(m), local_pf(i,j,nzb+1),            &
                                             surf_lsm%upward(m) )
             ENDDO
          ELSE
            IF ( .NOT. ALLOCATED( qsws_liq_av ) )  THEN
               ALLOCATE( qsws_liq_av(nysg:nyng,nxlg:nxrg) )
               qsws_liq_av = 0.0_wp
            ENDIF
             DO  i = nxl, nxr
                DO  j = nys, nyn
                   local_pf(i,j,nzb+1) = qsws_liq_av(j,i)
                ENDDO
             ENDDO
          ENDIF

          two_d = .TRUE.
          grid = 'zu1'

       CASE ( 'qsws_soil*_xy' )        ! 2d-array
          IF ( av == 0 )  THEN
             DO  m = 1, surf_lsm%ns
                i                   = surf_lsm%i(m)
                j                   = surf_lsm%j(m)
                local_pf(i,j,nzb+1) = MERGE( surf_lsm%qsws_soil(m), local_pf(i,j,nzb+1),           &
                                             surf_lsm%upward(m) )
             ENDDO
          ELSE
            IF ( .NOT. ALLOCATED( qsws_soil_av ) )  THEN
               ALLOCATE( qsws_soil_av(nysg:nyng,nxlg:nxrg) )
               qsws_soil_av = 0.0_wp
            ENDIF
             DO  i = nxl, nxr
                DO  j = nys, nyn
                   local_pf(i,j,nzb+1) = qsws_soil_av(j,i)
                ENDDO
             ENDDO
          ENDIF

          two_d = .TRUE.
          grid = 'zu1'

       CASE ( 'qsws_veg*_xy' )        ! 2d-array
          IF ( av == 0 )  THEN
             DO  m = 1, surf_lsm%ns
                i                   = surf_lsm%i(m)
                j                   = surf_lsm%j(m)
                local_pf(i,j,nzb+1) = MERGE( surf_lsm%qsws_veg(m), local_pf(i,j,nzb+1),            &
                                             surf_lsm%upward(m) )
             ENDDO
          ELSE
            IF ( .NOT. ALLOCATED( qsws_veg_av ) )  THEN
               ALLOCATE( qsws_veg_av(nysg:nyng,nxlg:nxrg) )
               qsws_veg_av = 0.0_wp
            ENDIF
             DO  i = nxl, nxr
                DO  j = nys, nyn
                   local_pf(i,j,nzb+1) = qsws_veg_av(j,i)
                ENDDO
             ENDDO
          ENDIF

          two_d = .TRUE.
          grid = 'zu1'


       CASE ( 'r_s*_xy' )        ! 2d-array
          IF ( av == 0 )  THEN
             DO  m = 1, surf_lsm%ns
                i                   = surf_lsm%i(m)
                j                   = surf_lsm%j(m)
                local_pf(i,j,nzb+1) = MERGE( surf_lsm%r_s(m), local_pf(i,j,nzb+1),                 &
                                             surf_lsm%upward(m) )
             ENDDO
          ELSE
            IF ( .NOT. ALLOCATED( r_s_av ) )  THEN
               ALLOCATE( r_s_av(nysg:nyng,nxlg:nxrg) )
               r_s_av = 0.0_wp
            ENDIF
             DO  i = nxl, nxr
                DO  j = nys, nyn
                   local_pf(i,j,nzb+1) = r_s_av(j,i)
                ENDDO
             ENDDO
          ENDIF

          two_d = .TRUE.
          grid = 'zu1'

       CASE ( 't_soil_xy', 't_soil_xz', 't_soil_yz' )
          IF ( av == 0 )  THEN
             DO  m = 1, surf_lsm%ns
                i   = surf_lsm%i(m)
                j   = surf_lsm%j(m)
                DO  k = nzb_soil, nzt_soil
                   local_pf(i,j,k) = MERGE( t_soil%var_2d(k,m), local_pf(i,j,k),                   &
                                            surf_lsm%upward(m) )
                ENDDO
             ENDDO
          ELSE
            IF ( .NOT. ALLOCATED( t_soil_av ) )  THEN
               ALLOCATE( t_soil_av(nzb_soil:nzt_soil,nysg:nyng,nxlg:nxrg) )
               t_soil_av = 0.0_wp
            ENDIF
             DO  i = nxl, nxr
                DO  j = nys, nyn
                   DO k = nzb_soil, nzt_soil
                      local_pf(i,j,k) = t_soil_av(k,j,i)
                   ENDDO
                ENDDO
             ENDDO
          ENDIF

          nzb_do = nzb_soil
          nzt_do = nzt_soil

          IF ( mode == 'xy' )  grid = 'zs'


       CASE DEFAULT
          found = .FALSE.
          grid  = 'none'

    END SELECT

 END SUBROUTINE lsm_data_output_2d


!--------------------------------------------------------------------------------------------------!
!
! Description:
! ------------
!> Subroutine defining 3D output variables
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_data_output_3d( av, variable, found, local_pf )


    USE indices


    IMPLICIT NONE

    CHARACTER (LEN=*) ::  variable !<

    INTEGER(iwp) ::  av    !<
    INTEGER(iwp) ::  i     !<
    INTEGER(iwp) ::  j     !<
    INTEGER(iwp) ::  k     !<
    INTEGER(iwp) ::  m     !< running index

    LOGICAL      ::  found !<

    REAL(wp), DIMENSION(nxl:nxr,nys:nyn,nzb_soil:nzt_soil) ::  local_pf !<


    found = .TRUE.


    SELECT CASE ( TRIM( variable ) )
!
!--   Requires 3D exchange

      CASE ( 'm_soil' )

         IF ( av == 0 )  THEN
            DO  m = 1, surf_lsm%ns
                i   = surf_lsm%i(m)
                j   = surf_lsm%j(m)
                DO  k = nzb_soil, nzt_soil
                   local_pf(i,j,k) = MERGE( m_soil%var_2d(k,m), local_pf(i,j,k), surf_lsm%upward(m) )
                ENDDO
            ENDDO
         ELSE
            IF ( .NOT. ALLOCATED( m_soil_av ) )  THEN
               ALLOCATE( m_soil_av(nzb_soil:nzt_soil,nysg:nyng,nxlg:nxrg) )
               m_soil_av = 0.0_wp
            ENDIF
            DO  i = nxl, nxr
               DO  j = nys, nyn
                  DO  k = nzb_soil, nzt_soil
                     local_pf(i,j,k) = m_soil_av(k,j,i)
                  ENDDO
               ENDDO
            ENDDO
         ENDIF

      CASE ( 't_soil' )

         IF ( av == 0 )  THEN
            DO  m = 1, surf_lsm%ns
               i   = surf_lsm%i(m)
               j   = surf_lsm%j(m)
               DO  k = nzb_soil, nzt_soil
                  local_pf(i,j,k) = MERGE( t_soil%var_2d(k,m), local_pf(i,j,k), surf_lsm%upward(m) )
               ENDDO
            ENDDO
         ELSE
            IF ( .NOT. ALLOCATED( t_soil_av ) )  THEN
               ALLOCATE( t_soil_av(nzb_soil:nzt_soil,nysg:nyng,nxlg:nxrg) )
               t_soil_av = 0.0_wp
            ENDIF
            DO  i = nxl, nxr
               DO  j = nys, nyn
                  DO  k = nzb_soil, nzt_soil
                     local_pf(i,j,k) = t_soil_av(k,j,i)
                  ENDDO
               ENDDO
            ENDDO
         ENDIF

       CASE DEFAULT
          found = .FALSE.

    END SELECT

 END SUBROUTINE lsm_data_output_3d


!--------------------------------------------------------------------------------------------------!
!
! Description:
! ------------
!> Calculate module specific statistics for land surface model, i.e. timeseries. Profiles could be
!> added in the future.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_statistics( mode, sr )

    CHARACTER (LEN=*) ::  mode  !< output mode

    INTEGER(iwp) ::  sr    !< number of statistic region


    IF ( mode == 'time_series' )  THEN
!
!--    Store time series date.
       ts_value(dots_start_index_lsm,sr)   = hom(nzb,1,93,sr)  ! ghf
       ts_value(dots_start_index_lsm+1,sr) = hom(nzb,1,94,sr)  ! qsws_liq
       ts_value(dots_start_index_lsm+2,sr) = hom(nzb,1,95,sr)  ! qsws_soil
       ts_value(dots_start_index_lsm+3,sr) = hom(nzb,1,96,sr)  ! qsws_veg
       ts_value(dots_start_index_lsm+4,sr) = hom(nzb,1,97,sr)  ! r_a
       ts_value(dots_start_index_lsm+5,sr) = hom(nzb,1,98,sr)  ! r_s
    ENDIF

 END SUBROUTINE lsm_statistics


!--------------------------------------------------------------------------------------------------!
!
! Description:
! ------------
!> Write restart data for land surface model.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_wrd_local


    IMPLICIT NONE

    INTEGER(idp),DIMENSION(nys:nyn,nxl:nxr) ::  global_end_index    !< end index for surface data (MPI-IO)
    INTEGER(idp),DIMENSION(nys:nyn,nxl:nxr) ::  global_start_index  !< start index for surface data (MPI-IO)

    LOGICAL ::  surface_data_to_write  !< switch for MPI-I/O if PE has surface data to write


    IF ( TRIM( restart_data_format_output ) == 'fortran_binary' )  THEN

       CALL wrd_write_string( 'ns_on_file_lsm' )
       WRITE ( 14 )  surf_lsm%ns

       IF ( ALLOCATED( c_liq_av ) )  THEN
          CALL wrd_write_string( 'c_liq_av' )
          WRITE ( 14 )  c_liq_av
       ENDIF

       IF ( ALLOCATED( c_soil_av ) )  THEN
          CALL wrd_write_string( 'c_soil_av' )
          WRITE ( 14 )  c_soil_av
       ENDIF

       IF ( ALLOCATED( c_veg_av ) )  THEN
          CALL wrd_write_string( 'c_veg_av' )
          WRITE ( 14 )  c_veg_av
       ENDIF

       IF ( ALLOCATED( lai_av ) )  THEN
          CALL wrd_write_string( 'lai_av' )
          WRITE ( 14 )  lai_av
       ENDIF

       IF ( ALLOCATED( m_liq_av ) )  THEN
          CALL wrd_write_string( 'm_liq_av' )
          WRITE ( 14 )  m_liq_av
       ENDIF

       IF ( ALLOCATED( m_soil_av ) )  THEN
          CALL wrd_write_string( 'm_soil_av' )
          WRITE ( 14 )  m_soil_av
       ENDIF

       IF ( ALLOCATED( qsws_liq_av ) )  THEN
          CALL wrd_write_string( 'qsws_liq_av' )
          WRITE ( 14 )  qsws_liq_av
       ENDIF

       IF ( ALLOCATED( qsws_soil_av ) )  THEN
          CALL wrd_write_string( 'qsws_soil_av' )
          WRITE ( 14 )  qsws_soil_av
       ENDIF

       IF ( ALLOCATED( qsws_veg_av ) )  THEN
          CALL wrd_write_string( 'qsws_veg_av' )
          WRITE ( 14 )  qsws_veg_av
       ENDIF

       IF ( ALLOCATED( t_soil_av ) )  THEN
          CALL wrd_write_string( 't_soil_av' )
          WRITE ( 14 )  t_soil_av
       ENDIF

       CALL wrd_write_string( 'lsm_start_index' )
       WRITE ( 14 )  surf_lsm%start_index

       CALL wrd_write_string( 'lsm_end_index' )
       WRITE ( 14 )  surf_lsm%end_index

       CALL wrd_write_string( 't_soil' )
       WRITE ( 14 )  t_soil%var_2d

       CALL wrd_write_string( 'm_soil' )
       WRITE ( 14 )  m_soil%var_2d

       CALL wrd_write_string( 'm_liq' )
       WRITE ( 14 )  m_liq%var_1d

       CALL wrd_write_string( 't_surface' )
       WRITE ( 14 )  t_surface%var_1d

    ELSEIF ( restart_data_format_output(1:3) == 'mpi' )  THEN

       IF ( ALLOCATED( c_liq_av )     )  CALL wrd_mpi_io( 'c_liq_av',  c_liq_av )
       IF ( ALLOCATED( c_soil_av )    )  CALL wrd_mpi_io( 'c_soil_av',  c_soil_av )
       IF ( ALLOCATED( c_veg_av )     )  CALL wrd_mpi_io( 'c_veg_av',  c_veg_av )
       IF ( ALLOCATED( lai_av )       )  CALL wrd_mpi_io( 'lai_av', lai_av )
       IF ( ALLOCATED( m_liq_av )     )  CALL wrd_mpi_io( 'm_liq_av',  m_liq_av )
       IF ( ALLOCATED( m_soil_av )    )  CALL wrd_mpi_io( 'm_soil_av', m_soil_av, nzb_soil, nzt_soil )
       IF ( ALLOCATED( qsws_liq_av )  )  CALL wrd_mpi_io( 'qsws_liq_av', qsws_liq_av )
       IF ( ALLOCATED( qsws_soil_av ) )  CALL wrd_mpi_io( 'qsws_soil_av', qsws_soil_av )
       IF ( ALLOCATED( qsws_veg_av )  )  CALL wrd_mpi_io( 'qsws_veg_av', qsws_veg_av )
       IF ( ALLOCATED( t_soil_av )    )  CALL wrd_mpi_io( 't_soil_av', t_soil_av, nzb_soil, nzt_soil )

       CALL rd_mpi_io_surface_filetypes( surf_lsm%start_index, surf_lsm%end_index,                 &
                                         surface_data_to_write, global_start_index,                &
                                         global_end_index )

       CALL wrd_mpi_io( 'lsm_global_start_index', global_start_index )
       CALL wrd_mpi_io( 'lsm_global_end_index', global_end_index )

       IF ( surface_data_to_write )  THEN
          CALL wrd_mpi_io_surface( 't_soil', t_soil%var_2d )
          CALL wrd_mpi_io_surface( 'm_soil', m_soil%var_2d )
          CALL wrd_mpi_io_surface( 'm_liq', m_liq%var_1d )
          CALL wrd_mpi_io_surface( 't_surface', t_surface%var_1d )
       ENDIF

    ENDIF

 END SUBROUTINE lsm_wrd_local


!--------------------------------------------------------------------------------------------------!
!
! Description:
! ------------
!> Read module-specific local restart data arrays (Fortran binary format).
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_rrd_local_ftn( k, nxlf, nxlc, nxl_on_file, nxrf, nxrc, nxr_on_file, nynf, nync,    &
                               nyn_on_file, nysf, nysc, nys_on_file, tmp_2d, found )


    USE control_parameters

    USE indices

    USE pegrid


    IMPLICIT NONE

    INTEGER(iwp) ::  k                 !<
    INTEGER(iwp) ::  nxlc              !<
    INTEGER(iwp) ::  nxlf              !<
    INTEGER(iwp) ::  nxl_on_file       !< index of left boundary on former local domain
    INTEGER(iwp) ::  nxrc              !<
    INTEGER(iwp) ::  nxrf              !<
    INTEGER(iwp) ::  nxr_on_file       !< index of right boundary on former local domain
    INTEGER(iwp) ::  nync              !<
    INTEGER(iwp) ::  nynf              !<
    INTEGER(iwp) ::  nyn_on_file       !< index of north boundary on former local domain
    INTEGER(iwp) ::  nysc              !<
    INTEGER(iwp) ::  nysf              !<
    INTEGER(iwp) ::  nys_on_file       !< index of south boundary on former local domain

    INTEGER(iwp) ::  ns_on_file_lsm    !< number of surface elements (natural type) on file

    INTEGER(iwp), DIMENSION(:,:), ALLOCATABLE, SAVE ::  start_index_on_file
    INTEGER(iwp), DIMENSION(:,:), ALLOCATABLE, SAVE ::  end_index_on_file

    LOGICAL, INTENT(OUT)  :: found

    REAL(wp), DIMENSION(nys_on_file-nbgp:nyn_on_file+nbgp,nxl_on_file-nbgp:nxr_on_file+nbgp) :: tmp_2d   !<

    REAL(wp), DIMENSION(nzb_soil:nzt_soil,nys_on_file-nbgp:nyn_on_file+nbgp,nxl_on_file-nbgp:nxr_on_file+nbgp) :: tmp_3d   !<

    TYPE(surf_type_lsm), SAVE :: tmp_walltype_1d   !< temporary 1D array containing the respective surface variable stored on file
    TYPE(surf_type_lsm), SAVE :: tmp_walltype_2d   !< temporary 2D array containing the respective surface variable stored on file,
    TYPE(surf_type_lsm), SAVE :: tmp_walltype_2d2  !< temporary 2D array containing the respective surface variable stored on file

    found = .TRUE.

    SELECT CASE ( restart_string(1:length) )

       CASE ( 'ns_on_file_lsm' )
          IF ( k == 1 )  THEN
             READ ( 13 )  ns_on_file_lsm
!
!--          In case of changing mpi topology, this routine could be called more than once.
!--          Hence, arrays need to be deallocated before allocated again.
             IF ( ALLOCATED( tmp_walltype_1d%var_1d )  )  DEALLOCATE( tmp_walltype_1d%var_1d )
             IF ( ALLOCATED( tmp_walltype_2d%var_2d )  )  DEALLOCATE( tmp_walltype_2d%var_2d )
             IF ( ALLOCATED( tmp_walltype_2d2%var_2d ) )  DEALLOCATE( tmp_walltype_2d2%var_2d )
!
!--          Allocate temporary arrays for reading data on file. Note, the size of allocated surface
!--          elements do not necessarily need to match the size of present surface elements on
!--          current processor, as the number of processors between restarts can change.
             ALLOCATE( tmp_walltype_1d%var_1d(1:ns_on_file_lsm) )
             ALLOCATE( tmp_walltype_2d%var_2d(nzb_soil:nzt_soil+1,1:ns_on_file_lsm) )
             ALLOCATE( tmp_walltype_2d2%var_2d(nzb_soil:nzt_soil,1:ns_on_file_lsm)  )
          ENDIF

       CASE ( 'c_liq_av' )
          IF ( .NOT. ALLOCATED( c_liq_av ) )  THEN
             ALLOCATE( c_liq_av(nysg:nyng,nxlg:nxrg) )
          ENDIF
          IF ( k == 1 )  READ ( 13 )  tmp_2d
          c_liq_av(nysc-nbgp:nync+nbgp,nxlc-nbgp:nxrc+nbgp) =                                      &
             tmp_2d(nysf-nbgp:nynf+nbgp,nxlf-nbgp:nxrf+nbgp)

       CASE ( 'c_soil_av' )
          IF ( .NOT. ALLOCATED( c_soil_av ) )  THEN
             ALLOCATE( c_soil_av(nysg:nyng,nxlg:nxrg) )
          ENDIF
          IF ( k == 1 )  READ ( 13 )  tmp_2d
          c_soil_av(nysc-nbgp:nync+nbgp,nxlc-nbgp:nxrc+nbgp) =                                     &
             tmp_2d(nysf-nbgp:nynf+nbgp,nxlf-nbgp:nxrf+nbgp)

       CASE ( 'c_veg_av' )
          IF ( .NOT. ALLOCATED( c_veg_av ) )  THEN
             ALLOCATE( c_veg_av(nysg:nyng,nxlg:nxrg) )
          ENDIF
          IF ( k == 1 )  READ ( 13 )  tmp_2d
          c_veg_av(nysc-nbgp:nync+nbgp,nxlc-nbgp:nxrc+nbgp) =                                      &
             tmp_2d(nysf-nbgp:nynf+nbgp,nxlf-nbgp:nxrf+nbgp)

       CASE ( 'lai_av' )
          IF ( .NOT. ALLOCATED( lai_av ) )  THEN
             ALLOCATE( lai_av(nysg:nyng,nxlg:nxrg) )
          ENDIF
          IF ( k == 1 )  READ ( 13 )  tmp_2d
          lai_av(nysc-nbgp:nync+nbgp,nxlc-nbgp:nxrc+nbgp) =                                        &
             tmp_2d(nysf-nbgp:nynf+nbgp,nxlf-nbgp:nxrf+nbgp)

       CASE ( 'm_liq_av' )
          IF ( .NOT. ALLOCATED( m_liq_av ) )  THEN
             ALLOCATE( m_liq_av(nysg:nyng,nxlg:nxrg) )
          ENDIF
          IF ( k == 1 )  READ ( 13 )  tmp_2d
          m_liq_av(nysc-nbgp:nync+nbgp,nxlc-nbgp:nxrc+nbgp) =                                      &
             tmp_2d(nysf-nbgp:nynf+nbgp,nxlf-nbgp:nxrf+nbgp)

       CASE ( 'm_soil_av' )
          IF ( .NOT. ALLOCATED( m_soil_av ) )  THEN
             ALLOCATE( m_soil_av(nzb_soil:nzt_soil,nysg:nyng,nxlg:nxrg) )
          ENDIF
          IF ( k == 1 )  READ ( 13 )  tmp_3d(:,:,:)
          m_soil_av(:,nysc-nbgp:nync+nbgp,nxlc-nbgp:nxrc+nbgp) =                                   &
             tmp_3d(nzb_soil:nzt_soil,nysf-nbgp:nynf+nbgp,nxlf-nbgp:nxrf+nbgp)

       CASE ( 'qsws_liq_av' )
          IF ( .NOT. ALLOCATED( qsws_liq_av ) )  THEN
             ALLOCATE( qsws_liq_av(nysg:nyng,nxlg:nxrg) )
          ENDIF
          IF ( k == 1 )  READ ( 13 )  tmp_2d
          qsws_liq_av(nysc-nbgp:nync+nbgp,nxlc-nbgp:nxrc+nbgp)  =                                  &
             tmp_2d(nysf-nbgp:nynf+nbgp,nxlf-nbgp:nxrf+nbgp)
       CASE ( 'qsws_soil_av' )
          IF ( .NOT. ALLOCATED( qsws_soil_av ) )  THEN
             ALLOCATE( qsws_soil_av(nysg:nyng,nxlg:nxrg) )
          ENDIF
          IF ( k == 1 )  READ ( 13 )  tmp_2d
          qsws_soil_av(nysc-nbgp:nync+nbgp,nxlc-nbgp:nxrc+nbgp)  =                                 &
             tmp_2d(nysf-nbgp:nynf+nbgp,nxlf-nbgp:nxrf+nbgp)

       CASE ( 'qsws_veg_av' )
          IF ( .NOT. ALLOCATED( qsws_veg_av ) )  THEN
             ALLOCATE( qsws_veg_av(nysg:nyng,nxlg:nxrg) )
          ENDIF
          IF ( k == 1 )  READ ( 13 )  tmp_2d
          qsws_veg_av(nysc-nbgp:nync+nbgp,nxlc-nbgp:nxrc+nbgp)  =                                  &
             tmp_2d(nysf-nbgp:nynf+nbgp,nxlf-nbgp:nxrf+nbgp)

       CASE ( 't_soil_av' )
          IF ( .NOT. ALLOCATED( t_soil_av ) )  THEN
             ALLOCATE( t_soil_av(nzb_soil:nzt_soil,nysg:nyng,nxlg:nxrg) )
          ENDIF
          IF ( k == 1 )  READ ( 13 )  tmp_3d(:,:,:)
          t_soil_av(:,nysc-nbgp:nync+nbgp,nxlc-nbgp:nxrc+nbgp) =                                   &
             tmp_3d(:,nysf-nbgp:nynf+nbgp,nxlf-nbgp:nxrf+nbgp)

       CASE ( 'lsm_start_index'  )
            IF ( k == 1 )  THEN
               IF ( ALLOCATED( start_index_on_file ) )  DEALLOCATE( start_index_on_file )
               ALLOCATE ( start_index_on_file(nys_on_file:nyn_on_file,nxl_on_file:nxr_on_file) )
               READ ( 13 )  start_index_on_file
            ENDIF

       CASE ( 'lsm_end_index' )
            IF ( k == 1 )  THEN
               IF ( ALLOCATED( end_index_on_file ) )  DEALLOCATE( end_index_on_file )
               ALLOCATE ( end_index_on_file(nys_on_file:nyn_on_file,nxl_on_file:nxr_on_file) )
               READ ( 13 )  end_index_on_file

            ENDIF

       CASE ( 't_soil' )
          IF ( k == 1 )  THEN
             IF ( .NOT. ALLOCATED( t_soil%var_2d ) )                                               &
                ALLOCATE( t_soil%var_2d(nzb_soil:nzt_soil+1,1:surf_lsm%ns) )
             READ ( 13 )  tmp_walltype_2d%var_2d
          ENDIF
          CALL surface_restore_elements( t_soil%var_2d, tmp_walltype_2d%var_2d,                    &
                                         surf_lsm%start_index, start_index_on_file,                &
                                         end_index_on_file, nxlc, nysc, nxlf, nxrf, nysf, nynf,    &
                                         nys_on_file, nyn_on_file, nxl_on_file, nxr_on_file )

       CASE ( 'm_soil' )
          IF ( k == 1 )  THEN
             IF ( .NOT. ALLOCATED( m_soil%var_2d ) )                                               &
                ALLOCATE( m_soil%var_2d(nzb_soil:nzt_soil,1:surf_lsm%ns) )
             READ ( 13 )  tmp_walltype_2d2%var_2d
          ENDIF
          CALL surface_restore_elements( m_soil%var_2d, tmp_walltype_2d2%var_2d,                   &
                                         surf_lsm%start_index, start_index_on_file,                &
                                         end_index_on_file, nxlc, nysc, nxlf, nxrf, nysf, nynf,    &
                                         nys_on_file, nyn_on_file, nxl_on_file, nxr_on_file )

       CASE ( 'm_liq' )
          IF ( k == 1 )  THEN
             IF ( .NOT. ALLOCATED( m_liq%var_1d ) )  ALLOCATE( m_liq%var_1d(1:surf_lsm%ns) )
             READ ( 13 )  tmp_walltype_1d%var_1d
          ENDIF
          CALL surface_restore_elements( m_liq%var_1d, tmp_walltype_1d%var_1d,                     &
                                         surf_lsm%start_index, start_index_on_file,                &
                                         end_index_on_file, nxlc, nysc, nxlf, nxrf, nysf, nynf,    &
                                         nys_on_file, nyn_on_file, nxl_on_file, nxr_on_file )

       CASE ( 't_surface' )
          IF ( k == 1 )  THEN
             IF ( .NOT. ALLOCATED( t_surface%var_1d ) )                                           &
                ALLOCATE( t_surface%var_1d(1:surf_lsm%ns) )
             READ ( 13 )  tmp_walltype_1d%var_1d
          ENDIF
          CALL surface_restore_elements( t_surface%var_1d, tmp_walltype_1d%var_1d,                 &
                                         surf_lsm%start_index, start_index_on_file,                &
                                         end_index_on_file, nxlc, nysc, nxlf, nxrf, nysf, nynf,    &
                                         nys_on_file, nyn_on_file, nxl_on_file, nxr_on_file )

       CASE DEFAULT

          found = .FALSE.

    END SELECT


 END SUBROUTINE lsm_rrd_local_ftn

!--------------------------------------------------------------------------------------------------!
!
! Description:
! ------------
!> Read module-specific local restart data arrays (MPI-IO).
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_rrd_local_mpi

    USE control_parameters

    USE indices

    USE pegrid


    IMPLICIT NONE

    INTEGER(idp), DIMENSION(nys:nyn,nxl:nxr) ::  global_end_index    !< end index for surface data (MPI-IO)
    INTEGER(idp), DIMENSION(nys:nyn,nxl:nxr) ::  global_start_index  !< start index for surface data (MPI-IO)

    LOGICAL ::  array_found
    LOGICAL ::  data_to_read = .FALSE.   !< switch to steer reading of surface restart data
!
!-- Read arrays needed for time-averaging. Note, this is skipped in case of cyclic_fill runs.
    IF ( .NOT. cyclic_fill_initialization )  THEN
       CALL rd_mpi_io_check_array( 'c_liq_av', found = array_found )
       IF ( array_found )  THEN
          IF ( .NOT. ALLOCATED( c_liq_av ) )  ALLOCATE( c_liq_av(nysg:nyng,nxlg:nxrg) )
          CALL rrd_mpi_io( 'c_liq_av',  c_liq_av )
       ENDIF

       CALL rd_mpi_io_check_array( 'c_soil_av', found = array_found )
       IF ( array_found )  THEN
          IF ( .NOT. ALLOCATED( c_soil_av ) )  ALLOCATE( c_soil_av(nysg:nyng,nxlg:nxrg) )
          CALL rrd_mpi_io( 'c_soil_av',  c_soil_av )
       ENDIF

       CALL rd_mpi_io_check_array( 'c_veg_av', found = array_found )
       IF ( array_found )  THEN
          IF ( .NOT. ALLOCATED( c_veg_av ) )  ALLOCATE( c_veg_av(nysg:nyng,nxlg:nxrg) )
          CALL rrd_mpi_io( 'c_veg_av',  c_veg_av )
       ENDIF

       CALL rd_mpi_io_check_array( 'lai_av', found = array_found )
       IF ( array_found )  THEN
          IF ( .NOT. ALLOCATED( lai_av ) )  ALLOCATE( lai_av(nysg:nyng,nxlg:nxrg) )
          CALL rrd_mpi_io( 'lai_av',  lai_av )
       ENDIF

       CALL rd_mpi_io_check_array( 'm_liq_av', found = array_found )
       IF ( array_found )  THEN
          IF ( .NOT. ALLOCATED( m_liq_av ) )  ALLOCATE( m_liq_av(nysg:nyng,nxlg:nxrg) )
          CALL rrd_mpi_io( 'm_liq_av',  m_liq_av )
       ENDIF

       CALL rd_mpi_io_check_array( 'm_soil_av', found = array_found )
       IF ( array_found )  THEN
          IF ( .NOT. ALLOCATED( m_soil_av ) )  ALLOCATE( m_soil_av(nzb_soil:nzt_soil,nysg:nyng,nxlg:nxrg) )
          CALL rrd_mpi_io( 'm_soil_av', m_soil_av, nzb_soil, nzt_soil )
       ENDIF

       CALL rd_mpi_io_check_array( 'qsws_liq_av', found = array_found )
       IF ( array_found )  THEN
          IF ( .NOT. ALLOCATED( qsws_liq_av ) )  ALLOCATE( qsws_liq_av(nysg:nyng,nxlg:nxrg) )
          CALL rrd_mpi_io( 'qsws_liq_av',  qsws_liq_av )
       ENDIF

       CALL rd_mpi_io_check_array( 'qsws_soil_av', found = array_found )
       IF ( array_found )  THEN
          IF ( .NOT. ALLOCATED( qsws_soil_av ) )  ALLOCATE( qsws_soil_av(nysg:nyng,nxlg:nxrg) )
          CALL rrd_mpi_io( 'qsws_soil_av',  qsws_soil_av )
       ENDIF

       CALL rd_mpi_io_check_array( 'qsws_veg_av', found = array_found )
       IF ( array_found )  THEN
          IF ( .NOT. ALLOCATED( qsws_veg_av ) )  ALLOCATE( qsws_veg_av(nysg:nyng,nxlg:nxrg) )
          CALL rrd_mpi_io( 'qsws_veg_av',  qsws_veg_av )
       ENDIF

       CALL rd_mpi_io_check_array( 't_soil_av', found = array_found )
       IF ( array_found )  THEN
          IF ( .NOT. ALLOCATED( t_soil_av ) )  ALLOCATE( t_soil_av(nzb_soil:nzt_soil,nysg:nyng,nxlg:nxrg) )
          CALL rrd_mpi_io( 't_soil_av', t_soil_av, nzb_soil, nzt_soil )
       ENDIF
    ENDIF
!
!-- Read surface-related data. First of all, check if LSM data is present on subdomain.
!-- At the moment reading of surface data in combination with cyclic fill is not realized,
!-- so that this is skipped for the moment.
    IF ( cyclic_fill_initialization )  RETURN

    CALL rd_mpi_io_check_array( 'lsm_global_start_index', found = array_found )
    IF ( array_found )  CALL rrd_mpi_io( 'lsm_global_start_index', global_start_index )

    CALL rd_mpi_io_check_array( 'lsm_global_end_index', found = array_found )
    IF ( array_found )  CALL rrd_mpi_io( 'lsm_global_end_index', global_end_index )
!
!-- Check if data input for surface-type variables is required. Note, only invoke routine if LSM
!-- surface restart data is on file. In case of cyclic fill initialization this is not necessarily
!-- guaranteed. To check this use the array_found control flag.
    IF ( array_found )  THEN
       CALL rd_mpi_io_surface_filetypes( surf_lsm%start_index, surf_lsm%end_index,                 &
                                         data_to_read, global_start_index, global_end_index )
    ENDIF

    IF ( data_to_read )  THEN
       CALL rrd_mpi_io_surface( 't_soil', t_soil%var_2d )
       CALL rrd_mpi_io_surface( 'm_soil',  m_soil%var_2d )
       CALL rrd_mpi_io_surface( 'm_liq', m_liq%var_1d )
       CALL rrd_mpi_io_surface( 't_surface', t_surface%var_1d )
    ENDIF

 END SUBROUTINE lsm_rrd_local_mpi


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Calculate maximum allowed timestep at LSM surfaces.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_timestep
!
!-- Consider a pre-factor (1/8) for the diffusion criterion.
    IF ( ALLOCATED( surf_lsm%dt_max ) )  dt_lsm = MINVAL( surf_lsm%dt_max ) * 0.125_wp
#if defined( __parallel )
    IF ( collective_wait )  CALL MPI_BARRIER( comm2d, ierr )
    CALL MPI_ALLREDUCE( MPI_IN_PLACE, dt_lsm, 1, MPI_REAL, MPI_MIN, comm2d, ierr )
#endif

 END SUBROUTINE lsm_timestep


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Sampling of LSM variables along customized measurement coordinates. Surface variables are
!> sampled on the atmosphere index space, while soil variables are sampled on the soil index space.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE lsm_vm_sampling( variable, var_atmos, i_atmos, j_atmos, k_atmos, ns_atmos,             &
                             var_soil, i_soil, j_soil, k_soil, ns_soil, sampled )

    CHARACTER(LEN=*) ::  variable  !< treated variable

    INTEGER(iwp) ::  i         !< grid index in x-direction
    INTEGER(iwp) ::  j         !< grid index in y-direction
    INTEGER(iwp) ::  k         !< grid index in z-direction
    INTEGER(iwp) ::  m         !< running index over all virtual observation coordinates
    INTEGER(iwp) ::  mm        !< index of surface element which corresponds to the virtual observation coordinate
    INTEGER(iwp) ::  ns_atmos  !< number of sampling points for atmosphere and surface variables
    INTEGER(iwp) ::  ns_soil   !< number of sampling points for soil variables

    INTEGER(iwp), DIMENSION(1:ns_atmos) ::  i_atmos  !< sampling index in x-direction for atmosphere variables
    INTEGER(iwp), DIMENSION(1:ns_atmos) ::  j_atmos  !< sampling index in y-direction for atmosphere variables
    INTEGER(iwp), DIMENSION(1:ns_atmos) ::  k_atmos  !< sampling index in z-direction for atmosphere variables

    INTEGER(iwp), DIMENSION(1:ns_soil) ::  i_soil  !< sampling index in x-direction for soil variables
    INTEGER(iwp), DIMENSION(1:ns_soil) ::  j_soil  !< sampling index in y-direction for soil variables
    INTEGER(iwp), DIMENSION(1:ns_soil) ::  k_soil  !< sampling index in z-direction for soil variables

    LOGICAL ::  sampled !< flag indicating whether a variable has been sampled

    REAL(wp), DIMENSION(1:ns_atmos) ::  var_atmos  !< array to store atmosphere variables

    REAL(wp), DIMENSION(1:ns_soil) ::  var_soil   !< array to store soil variables


    SELECT CASE ( TRIM( variable ) )
!
!--    Ground heat flux.
       CASE ( 'hfdg' )
          DO  m = 1, ns_atmos
!
!--          Only sample surface data which is defined on the subdomain.
             IF ( j_atmos(m) >= nys  .AND.  j_atmos(m) <= nyn  .AND.                               &
                  i_atmos(m) >= nxl  .AND.  i_atmos(m) <= nxr )                                    &
             THEN
                j = j_atmos(m)
                i = i_atmos(m)
!
!--             Take only values from horizontally-upward facing surfaces.
                DO  mm = surf_lsm%start_index(j,i), surf_lsm%end_index(j,i)
                   var_atmos(m) = MERGE( surf_lsm%ghf(mm), var_atmos(m), surf_lsm%upward(mm) )
                ENDDO
             ENDIF
          ENDDO
          sampled = .TRUE.
!
!--    Soil moisture.
       CASE ( 'm_soil', 'lwcs' )
          DO  m = 1, ns_soil
             IF ( j_soil(m) >= nys  .AND.  j_soil(m) <= nyn  .AND.                                 &
                  i_soil(m) >= nxl  .AND.  i_soil(m) <= nxr )                                      &
             THEN
                k = k_soil(m)
                j = j_soil(m)
                i = i_soil(m)
!
!--             Take only values from horizontally-upward facing surfaces.
                DO  mm = surf_lsm%start_index(j,i), surf_lsm%end_index(j,i)
                   var_soil(m) = MERGE( m_soil%var_2d(k,mm), var_soil(m), surf_lsm%upward(mm) )
                ENDDO
             ENDIF
          ENDDO
          sampled = .TRUE.
!
!--    Water temperature.
       CASE ( 't_lw' )
          DO  m = 1, ns_atmos
!
!--          Only sample surface data which is defined on the subdomain.
             IF ( j_atmos(m) >= nys  .AND.  j_atmos(m) <= nyn  .AND.                               &
                  i_atmos(m) >= nxl  .AND.  i_atmos(m) <= nxr )                                    &
             THEN
                j = j_atmos(m)
                i = i_atmos(m)
!
!--             Take only values from horizontally-upward facing surfaces.
                DO  mm = surf_lsm%start_index(j,i), surf_lsm%end_index(j,i)
                   var_atmos(m) = MERGE( t_soil%var_2d(nzb_soil,m), var_atmos(m),                  &
                                         surf_lsm%upward(mm) )
                ENDDO
             ENDIF
          ENDDO
          sampled = .TRUE.
!
!--    Soil and wall temperature.
       CASE ( 't_soil' )
          DO  m = 1, ns_soil
             IF ( j_soil(m) >= nys  .AND.  j_soil(m) <= nyn  .AND.                                 &
                  i_soil(m) >= nxl  .AND.  i_soil(m) <= nxr )                                      &
             THEN
                k = k_soil(m)
                j = j_soil(m)
                i = i_soil(m)
!
!--             Take only values from horizontally-upward facing surfaces.
                DO  mm = surf_lsm%start_index(j,i), surf_lsm%end_index(j,i)
                   var_soil(m) = MERGE( t_soil%var_2d(k,mm), var_soil(m), surf_lsm%upward(mm) )
                ENDDO
             ENDIF
          ENDDO
          sampled = .TRUE.

       CASE DEFAULT

    END SELECT
!
!-- Avoid compiler warning for unused variables by constructing an if condition which is never
!-- fulfilled.
    IF ( .FALSE.  .AND.  ns_atmos < 0  .AND.  ns_soil < 0 )  THEN
       k_atmos = k_atmos
    ENDIF

 END SUBROUTINE lsm_vm_sampling


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Calculation of roughness length for open water (lakes, ocean). The parameterization follows
!> Charnock (1955). Two different implementations are available: as in ECMWF-IFS (Beljaars 1994) or
!> as in FLake (Subin et al. 2012)
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE calc_z0_water_surface

    USE control_parameters,                                                                        &
        ONLY:  message_string,                                                                     &
               molecular_viscosity

    INTEGER(iwp) ::  m       !< running index

    LOGICAL ::  flag_exceed_z0   !< dummy flag to indicate whether roughness length is too high
    LOGICAL ::  flag_exceed_z0h  !< dummy flag to indicate whether roughness length for temperature is too high
    LOGICAL ::  flag_exceed_z0q  !< dummy flag to indicate whether roughness length for mositure is too high

    REAL(wp), PARAMETER :: alpha_ch  = 0.018_wp !< Charnock constant (0.01-0.11). Use 0.01 for FLake and 0.018 for ECMWF
!       REAL(wp), PARAMETER :: pr_number = 0.71_wp !< molecular Prandtl number in the Charnock parameterization (differs from
!                                                  !< prandtl_number)
!       REAL(wp), PARAMETER :: sc_number = 0.66_wp !< molecular Schmidt number in the Charnock parameterization
!       REAL(wp) :: re_0 !< near-surface roughness Reynolds number


    flag_exceed_z0  = .FALSE.
    flag_exceed_z0h = .FALSE.
    flag_exceed_z0q = .FALSE.
    !$OMP PARALLEL DO PRIVATE (m) SCHEDULE (STATIC)                                      &
    !$OMP&            REDUCTION(.OR.:flag_exceed_z0, flag_exceed_z0h)
    DO  m = 1, surf_lsm%ns
!
!--    Only upward facing horizontal surfaces are considered for water surface processing
       IF ( surf_lsm%water_surface(m)  .AND.  surf_lsm%upward(m) )  THEN
!           i   = surf_lsm%i(m)
!           j   = surf_lsm%j(m)
!
!!--       Disabled: FLake parameterization. Ideally, the Charnock coefficient should depend on
!!--       the water depth and the fetch length
!          re_0 = z0(j,i) * us(j,i) / molecular_viscosity
!
!          z0(j,i) = MAX( 0.1_wp * molecular_viscosity / us(j,i),            &
!                           alpha_ch * us(j,i) / g )
!
!          z0h(j,i) = z0(j,i) * EXP( - kappa / pr_number * ( 4.0_wp * SQRT( re_0 ) - 3.2_wp ) )
!          z0q(j,i) = z0(j,i) * EXP( - kappa / pr_number * ( 4.0_wp * SQRT( re_0 ) - 4.2_wp ) )

!
!          Set minimum roughness length for u* > 0.2
!          IF ( us(j,i) > 0.2_wp )  THEN
!             z0h(j,i) = MAX( 1.0E-5_wp, z0h(j,i) )
!             z0q(j,i) = MAX( 1.0E-5_wp, z0q(j,i) )
!          ENDIF

!
!--       ECMWF IFS model parameterization after Beljaars (1994). At low wind speed, the sea surface
!--       becomes aerodynamically smooth and the roughness scales with the viscosity. At high wind
!--       speed, the Charnock relation is used. Add a security factor of 1E-8 to avoid divisions by
!--       zero. Note, this is only applied for upward-facing surfaces.
          surf_lsm%z0(m)  = 0.11_wp * molecular_viscosity / ( surf_lsm%us(m) + 1.0E-8_wp ) +       &
                            alpha_ch * surf_lsm%us(m)**2 / g

          surf_lsm%z0h(m) = 0.40_wp * molecular_viscosity / ( surf_lsm%us(m) + 1.0E-8_wp )
          surf_lsm%z0q(m) = 0.62_wp * molecular_viscosity / ( surf_lsm%us(m) + 1.0E-8_wp )


!
!--       Check if roughness length for momentum, heat, or moisture exceed surface-layer height and
!--       limit local roughness length where necessary (if allowed). Error/informative messages
!--       are given only once in order to avoid the job protocol to be messed-up with messages.
          IF ( surf_lsm%z0(m) > 0.5_wp * surf_lsm%z_mo(m) )  THEN
             flag_exceed_z0 = .TRUE.
             IF ( allow_roughness_limitation )  surf_lsm%z0(m) = 0.5_wp * surf_lsm%z_mo(m)
          ENDIF

          IF ( surf_lsm%z0h(m) >= 0.5_wp * surf_lsm%z_mo(m) )  THEN
             flag_exceed_z0h = .TRUE.
             IF ( allow_roughness_limitation )  surf_lsm%z0h(m) = 0.5_wp * surf_lsm%z_mo(m)
          ENDIF

          IF ( surf_lsm%z0q(m) >= 0.5_wp * surf_lsm%z_mo(m) )  THEN
             flag_exceed_z0q = .TRUE.
             IF ( allow_roughness_limitation )  surf_lsm%z0q(m) = 0.5_wp * surf_lsm%z_mo(m)
          ENDIF

       ENDIF
    ENDDO

#if defined( __parallel )
    CALL MPI_ALLREDUCE( MPI_IN_PLACE, flag_exceed_z0, 1, MPI_LOGICAL, MPI_LOR, comm2d, ierr)
#endif
    IF ( flag_exceed_z0 )  THEN
       WRITE( message_string, * ) 'z0 exceeds 0.5 * surface-layer height at sea surface(s)'
       IF ( allow_roughness_limitation )  THEN
          message_string = TRIM( message_string ) // ' and is limited to that height'
          CALL message( 'calc_z0_water_surface', 'LSM0049', 0, 0, 0, 6, 0 )
       ELSE
          CALL message( 'calc_z0_water_surface', 'LSM0048', 2, 2, 0, 6, 0 )
       ENDIF
    ENDIF

#if defined( __parallel )
    CALL MPI_ALLREDUCE( MPI_IN_PLACE, flag_exceed_z0h, 1, MPI_LOGICAL, MPI_LOR, comm2d, ierr)
#endif
    IF ( flag_exceed_z0h )  THEN
       WRITE( message_string, * ) 'z0h exceeds 0.5 * surface-layer height at sea surface(s)'
       IF ( allow_roughness_limitation )  THEN
          message_string = TRIM( message_string ) // ' and is limited to that height'
          CALL message( 'calc_z0_water_surface', 'LSM0049', 0, 0, 0, 6, 0 )
       ELSE
          CALL message( 'calc_z0_water_surface', 'LSM0048', 2, 2, 0, 6, 0 )
       ENDIF
    ENDIF

#if defined( __parallel )
    CALL MPI_ALLREDUCE( MPI_IN_PLACE, flag_exceed_z0q, 1, MPI_LOGICAL, MPI_LOR, comm2d, ierr)
#endif
    IF ( flag_exceed_z0q )  THEN
       WRITE( message_string, * ) 'z0q exceeds 0.5 * surface-layer height at sea surface(s)'
       IF ( allow_roughness_limitation )  THEN
          message_string = TRIM( message_string ) // ' and is limited to that height'
          CALL message( 'calc_z0_water_surface', 'LSM0049', 0, 0, 0, 6, 0 )
       ELSE
          CALL message( 'calc_z0_water_surface', 'LSM0048', 2, 2, 0, 6, 0 )
       ENDIF
    ENDIF

 END SUBROUTINE calc_z0_water_surface


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!>  Vertical interpolation and extrapolation of 1D soil profile input from dynamic input file onto
!> the numeric vertical soil grid.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE interpolate_soil_profile( var, var_file, z_grid, z_file, nzb_var, nzt_var, nzb_file,   &
                                      nzt_file )

    IMPLICIT NONE

    INTEGER(iwp) ::  k        !< running index z-direction file
    INTEGER(iwp) ::  kk       !< running index z-direction stretched model grid
    INTEGER(iwp) ::  ku       !< upper index bound along z-direction for varialbe from file
    INTEGER(iwp) ::  nzb_var  !< lower bound of final array
    INTEGER(iwp) ::  nzt_var  !< upper bound of final array
    INTEGER(iwp) ::  nzb_file !< lower bound of file array
    INTEGER(iwp) ::  nzt_file !< upper bound of file array

    REAL(wp), DIMENSION(nzb_var:nzt_var)   ::  var      !< treated variable
    REAL(wp), DIMENSION(nzb_file:nzt_file) ::  var_file !< temporary variable
    REAL(wp), DIMENSION(nzb_file:nzt_file) ::  z_file   !< grid levels on file grid
    REAL(wp), DIMENSION(nzb_var:nzt_var)   ::  z_grid   !< grid levels on numeric grid


    ku = nzt_file

    DO  k = nzb_var, nzt_var
!
!--    Determine index on dynamic file grid which is closest to the actual height.
       kk = MINLOC( ABS( z_file - z_grid(k) ), DIM = 1 )
!
!--    If closest index on dynamic file grid is smaller than top index, interpolate the data.
       IF ( kk < nzt_file )  THEN
          IF ( z_file(kk) - z_grid(k) <= 0.0_wp )  THEN
             var(k) = var_file(kk) + ( var_file(kk+1) - var_file(kk) ) /                           &
                                     ( z_file(kk+1)   - z_file(kk)   ) *                           &
                                     ( z_grid(k)      - z_file(kk)   )

          ELSEIF ( z_file(kk) - z_grid(k) > 0.0_wp )  THEN
             var(k) = var_file(kk-1) + ( var_file(kk) - var_file(kk-1) ) /                         &
                                       ( z_file(kk)   - z_file(kk-1)   ) *                         &
                                       ( z_grid(k)    - z_file(kk-1)   )
          ENDIF
!
!--    Extrapolate if actual height is above the highest dynamic file grid level by the last value.
       ELSE
          var(k) = var_file(ku)
       ENDIF

    ENDDO

 END SUBROUTINE interpolate_soil_profile

 END MODULE land_surface_model_mod
