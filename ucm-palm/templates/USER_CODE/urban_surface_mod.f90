!> @file urban_surface_mod.f90
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
! Copyright 2015-2021 Czech Technical University in Prague
! Copyright 2015-2021 Institute of Computer Science of the Czech Academy of Sciences, Prague
! Copyright 1997-2021 Leibniz Universitaet Hannover
!--------------------------------------------------------------------------------------------------!
!
!
! Description:
! ------------
! 2016/6/9 - Initial version of the USM (Urban Surface Model)
!            authors: Jaroslav Resler, Pavel Krc (Czech Technical University in Prague and Institute
!            of Computer Science of the Czech Academy of Sciences, Prague)
!            with contributions: Michal Belda, Nina Benesova, Ondrej Vlcek
!            partly inspired by PALM LSM (B. Maronga)
!            parameterizations of Ra checked with TUF3D (E. S. Krayenhoff)
!> Module for Urban Surface Model (USM)
!> The module includes:
!>    1. Radiation model with direct/diffuse radiation, shading, reflections and integration with
!>       plant canopy
!>    2. Wall and wall surface model
!>    3. Surface layer energy balance
!>    4. Anthropogenic heat (only from transportation so far)
!>    5. Necessary auxiliary subroutines (reading inputs, writing outputs, restart simulations, ...)
!> It also makes use of standard radiation and integrates it into urban surface model.
!>
!> Further work:
!> -------------
!> @todo Revise initialization when building_pars / building_surface_pars are provided -
!>       intialization is not consistent to building_pars
!> @todo Revise flux conversion in energy-balance solver
!--------------------------------------------------------------------------------------------------!
 MODULE urban_surface_mod

#if defined( __parallel )
    USE MPI
#endif

    USE arrays_3d,                                                                                 &
        ONLY:  exner,                                                                              &
               hyp,                                                                                &
               hyrho,                                                                              &
               p,                                                                                  &
               prr,                                                                                &
               pt,                                                                                 &
               q,                                                                                  &
               ql,                                                                                 &
               tend,                                                                               &
               u,                                                                                  &
               v,                                                                                  &
               vpt,                                                                                &
               w,                                                                                  &
               zu

    USE calc_mean_profile_mod,                                                                     &
        ONLY:  calc_mean_profile

    USE basic_constants_and_equations_mod,                                                         &
        ONLY:  c_p,                                                                                &
               degc_to_k,                                                                          &
               g,                                                                                  &
               kappa,                                                                              &
               l_v,                                                                                &
               magnus_tl,                                                                          &
               pi,                                                                                 &
               r_d,                                                                                &
               rho_l,                                                                              &
               sigma_sb

    USE control_parameters,                                                                        &
        ONLY:  allow_roughness_limitation,                                                         &
               average_count_3d,                                                                   &
               coupling_char,                                                                      &
               coupling_start_time,                                                                &
               cyclic_fill_initialization,                                                         &
               debug_output,                                                                       &
               debug_output_timestep,                                                              &
               debug_string,                                                                       &
               dt_do3d,                                                                            &
               dt_3d,                                                                              &
               dz,                                                                                 &
               end_time,                                                                           &
               external_anthropogenic_heat,                                                        &
               humidity,                                                                           &
               indoor_model,                                                                       &
               initializing_actions,                                                               &
               intermediate_timestep_count,                                                        &
               intermediate_timestep_count_max,                                                    &
               io_blocks,                                                                          &
               io_group,                                                                           &
               large_scale_forcing,                                                                &
               lsf_surf,                                                                           &
               message_string,                                                                     &
               output_fill_value,                                                                  &
               pt_surface,                                                                         &
               read_spinup_data,                                                                   &
               restart_data_format_output,                                                         &
               surface_pressure,                                                                   &
               time_since_reference_point,                                                         &
               timestep_scheme,                                                                    &
               topography,                                                                         &
               tsc,                                                                                &
               urban_surface,                                                                      &
               varnamelength


    USE bulk_cloud_model_mod,                                                                      &
        ONLY:  bulk_cloud_model,                                                                   &
               precipitation

    USE cpulog,                                                                                    &
        ONLY:  cpu_log,                                                                            &
               log_point,                                                                          &
               log_point_s

    USE grid_variables,                                                                            &
        ONLY:  ddx,                                                                                &
               ddx2,                                                                               &
               ddy,                                                                                &
               ddy2,                                                                               &
               dx,                                                                                 &
               dy

    USE indices,                                                                                   &
        ONLY:  nbgp,                                                                               &
               nnx,                                                                                &
               nny,                                                                                &
               nnz,                                                                                &
               nx,                                                                                 &
               nxl,                                                                                &
               nxlg,                                                                               &
               nxr,                                                                                &
               nxrg,                                                                               &
               ny,                                                                                 &
               nyn,                                                                                &
               nyng,                                                                               &
               nys,                                                                                &
               nysg,                                                                               &
               nzb,                                                                                &
               nzt,                                                                                &
               topo_top_ind

    USE, INTRINSIC :: iso_c_binding

    USE kinds

    USE netcdf_data_input_mod,                                                                     &
        ONLY:  albedo_type_f,                                                                      &
               building_pars_f,                                                                    &
               building_surface_pars_f,                                                            &
               building_type_f,                                                                    &
               terrain_height_f

    USE palm_date_time_mod,                                                                        &
        ONLY:  get_date_time,                                                                      &
               seconds_per_hour

    USE pegrid

    USE radiation_model_mod,                                                                       &
        ONLY:  albedo_type,                                                                        &
               dirname,                                                                            &
               diridx,                                                                             &
               dirint,                                                                             &
               force_radiation_call,                                                               &
               id,                                                                                 &
               idown,                                                                              &
               ieast,                                                                              &
               inorth,                                                                             &
               isouth,                                                                             &
               iup,                                                                                &
               iwest,                                                                              &
               nd,                                                                                 &
               nz_urban_b,                                                                         &
               nz_urban_t,                                                                         &
               radiation_interaction,                                                              &
               radiation,                                                                          &
               rad_lw_in,                                                                          &
               rad_lw_out,                                                                         &
               rad_sw_in,                                                                          &
               rad_sw_out,                                                                         &
               unscheduled_radiation_calls

    USE restart_data_mpi_io_mod,                                                                   &
        ONLY:  rd_mpi_io_check_array,                                                              &
               rd_mpi_io_surface_filetypes,                                                        &
               rrd_mpi_io,                                                                         &
               rrd_mpi_io_surface,                                                                 &
               wrd_mpi_io,                                                                         &
               wrd_mpi_io_surface

    USE statistics,                                                                                &
        ONLY:  hom,                                                                                &
               statistic_regions

    USE surface_mod,                                                                               &
        ONLY:  ind_pav_green,                                                                      &
               ind_veg_wall,                                                                       &
               ind_wat_win,                                                                        &
               surf_type,                                                                          &
               surf_usm,                                                                           &
               surface_restore_elements

    IMPLICIT NONE

!
!-- USM model constants

    REAL(wp), PARAMETER ::  b_ch               = 6.04_wp    !< Clapp & Hornberger exponent
    REAL(wp), PARAMETER ::  lambda_h_green_dry = 0.19_wp    !< heat conductivity for dry soil
    REAL(wp), PARAMETER ::  lambda_h_green_sm  = 3.44_wp    !< heat conductivity of the soil matrix
    REAL(wp), PARAMETER ::  lambda_h_water     = 0.57_wp    !< heat conductivity of water
    REAL(wp), PARAMETER ::  psi_sat            = -0.388_wp  !< soil matrix potential at saturation
    REAL(wp), PARAMETER ::  rho_c_soil         = 2.19E6_wp  !< volumetric heat capacity of soil
    REAL(wp), PARAMETER ::  rho_c_water        = 4.20E6_wp  !< volumetric heat capacity of water
!    REAL(wp), PARAMETER ::  m_max_depth        = 0.0002_wp  !< Maximum capacity of the water reservoir (m)

!
!-- Soil parameters I           alpha_vg,      l_vg_green,    n_vg, gamma_w_green_sat
    REAL(wp), DIMENSION(0:3,1:7), PARAMETER ::  soil_pars = RESHAPE( (/     &
                                 3.83_wp,  1.250_wp, 1.38_wp,  6.94E-6_wp, &  !< soil 1
                                 3.14_wp, -2.342_wp, 1.28_wp,  1.16E-6_wp, &  !< soil 2
                                 0.83_wp, -0.588_wp, 1.25_wp,  0.26E-6_wp, &  !< soil 3
                                 3.67_wp, -1.977_wp, 1.10_wp,  2.87E-6_wp, &  !< soil 4
                                 2.65_wp,  2.500_wp, 1.10_wp,  1.74E-6_wp, &  !< soil 5
                                 1.30_wp,  0.400_wp, 1.20_wp,  0.93E-6_wp, &  !< soil 6
                                 0.00_wp,  0.00_wp,  0.00_wp,  0.57E-6_wp  &  !< soil 7
                                 /), (/ 4, 7 /) )

!
!-- Soil parameters II              swc_sat,     fc,   wilt,    swc_res
    REAL(wp), DIMENSION(0:3,1:7), PARAMETER ::  m_soil_pars = RESHAPE( (/ &
                                 0.403_wp, 0.244_wp, 0.059_wp, 0.025_wp, &  !< soil 1
                                 0.439_wp, 0.347_wp, 0.151_wp, 0.010_wp, &  !< soil 2
                                 0.430_wp, 0.383_wp, 0.133_wp, 0.010_wp, &  !< soil 3
                                 0.520_wp, 0.448_wp, 0.279_wp, 0.010_wp, &  !< soil 4
                                 0.614_wp, 0.541_wp, 0.335_wp, 0.010_wp, &  !< soil 5
                                 0.766_wp, 0.663_wp, 0.267_wp, 0.010_wp, &  !< soil 6
                                 0.472_wp, 0.323_wp, 0.171_wp, 0.000_wp  &  !< soil 7
                                 /), (/ 4, 7 /) )
!
!-- Value 9999999.9_wp -> Generic available or user-defined value must be set otherwise
!-- -> No generic variable and user setting is optional
    REAL(wp) ::  alpha_vangenuchten = 9999999.9_wp      !< NAMELIST alpha_vg
    REAL(wp) ::  field_capacity = 9999999.9_wp          !< NAMELIST fc
    REAL(wp) ::  hydraulic_conductivity = 9999999.9_wp  !< NAMELIST gamma_w_green_sat
    REAL(wp) ::  l_vangenuchten = 9999999.9_wp          !< NAMELIST l_vg
    REAL(wp) ::  n_vangenuchten = 9999999.9_wp          !< NAMELIST n_vg
    REAL(wp) ::  residual_moisture = 9999999.9_wp       !< NAMELIST m_res
    REAL(wp) ::  saturation_moisture = 9999999.9_wp     !< NAMELIST m_sat
    REAL(wp) ::  wilting_point = 9999999.9_wp           !< NAMELIST m_wilt

!
!-- Configuration parameters (they can be setup in PALM config)
    LOGICAL ::  force_radiation_call_l = .FALSE.   !< flag parameter for unscheduled radiation model calls
    LOGICAL ::  usm_wall_mod = .FALSE.             !< reduces conductivity of the first 2 wall layers by factor 0.1


    INTEGER(iwp) ::  building_type = 1               !< default building type (preleminary setting)
    INTEGER(iwp) ::  roof_category = 2               !< default category for root surface
    INTEGER(iwp) ::  wall_category = 2               !< default category for wall surface over pedestrian zone

    REAL(wp)     ::  d_roughness_concrete            !< inverse roughness length of average concrete surface
    REAL(wp)     ::  roughness_concrete = 0.001_wp   !< roughness length of average concrete surface

!
!-- Indices of input attributes in building_pars for (above) ground floor level
    INTEGER(iwp) ::  ind_alb_wall_agfl     = 38   !< index in input list for albedo_type of wall above ground floor level
    INTEGER(iwp) ::  ind_alb_wall_gfl      = 66   !< index in input list for albedo_type of wall ground floor level
    INTEGER(iwp) ::  ind_alb_wall_r        = 101  !< index in input list for albedo_type of wall roof
    INTEGER(iwp) ::  ind_alb_green_agfl    = 39   !< index in input list for albedo_type of green above ground floor level
    INTEGER(iwp) ::  ind_alb_green_gfl     = 78   !< index in input list for albedo_type of green ground floor level
    INTEGER(iwp) ::  ind_alb_green_r       = 117  !< index in input list for albedo_type of green roof
    INTEGER(iwp) ::  ind_alb_win_agfl      = 40   !< index in input list for albedo_type of window fraction above ground floor
                                                  !< level
    INTEGER(iwp) ::  ind_alb_win_gfl       = 77   !< index in input list for albedo_type of window fraction ground floor level
    INTEGER(iwp) ::  ind_alb_win_r         = 115  !< index in input list for albedo_type of window fraction roof
    INTEGER(iwp) ::  ind_emis_wall_agfl    = 14   !< index in input list for wall emissivity, above ground floor level
    INTEGER(iwp) ::  ind_emis_wall_gfl     = 32   !< index in input list for wall emissivity, ground floor level
    INTEGER(iwp) ::  ind_emis_wall_r       = 100  !< index in input list for wall emissivity, roof
    INTEGER(iwp) ::  ind_emis_green_agfl   = 15   !< index in input list for green emissivity, above ground floor level
    INTEGER(iwp) ::  ind_emis_green_gfl    = 34   !< index in input list for green emissivity, ground floor level
    INTEGER(iwp) ::  ind_emis_green_r      = 116  !< index in input list for green emissivity, roof
    INTEGER(iwp) ::  ind_emis_win_agfl     = 16   !< index in input list for window emissivity, above ground floor level
    INTEGER(iwp) ::  ind_emis_win_gfl      = 33   !< index in input list for window emissivity, ground floor level
    INTEGER(iwp) ::  ind_emis_win_r        = 113  !< index in input list for window emissivity, roof
    INTEGER(iwp) ::  ind_gflh              = 20   !< index in input list for ground floor level height
    INTEGER(iwp) ::  ind_green_frac_w_agfl = 2    !< index in input list for green fraction on wall, above ground floor level
    INTEGER(iwp) ::  ind_green_frac_w_gfl  = 23   !< index in input list for green fraction on wall, ground floor level
    INTEGER(iwp) ::  ind_green_frac_r_agfl = 3    !< index in input list for green fraction on roof, above ground floor level
    INTEGER(iwp) ::  ind_green_frac_r_gfl  = 24   !< index in input list for green fraction on roof, ground floor level
    INTEGER(iwp) ::  ind_green_type_roof   = 118  !< index in input list for type of green roof
    INTEGER(iwp) ::  ind_hc1_agfl          = 6    !< index in input list for heat capacity at first wall layer,
                                                  !< above ground floor level
    INTEGER(iwp) ::  ind_hc1_gfl           = 26   !< index in input list for heat capacity at first wall layer, ground floor level
    INTEGER(iwp) ::  ind_hc1_wall_r        = 94   !< index in input list for heat capacity at first wall layer, roof
    INTEGER(iwp) ::  ind_hc1_win_agfl      = 83   !< index in input list for heat capacity at first window layer,
                                                  !< above ground floor level
    INTEGER(iwp) ::  ind_hc1_win_gfl       = 71   !< index in input list for heat capacity at first window layer,
                                                  !< ground floor level
    INTEGER(iwp) ::  ind_hc1_win_r         = 107  !< index in input list for heat capacity at first window layer, roof
    INTEGER(iwp) ::  ind_hc2_agfl          = 7    !< index in input list for heat capacity at second wall layer,
                                                  !< above ground floor level
    INTEGER(iwp) ::  ind_hc2_gfl           = 27   !< index in input list for heat capacity at second wall layer, ground floor level
    INTEGER(iwp) ::  ind_hc2_wall_r        = 95   !< index in input list for heat capacity at second wall layer, roof
    INTEGER(iwp) ::  ind_hc2_win_agfl      = 84   !< index in input list for heat capacity at second window layer,
                                                  !< above ground floor level
    INTEGER(iwp) ::  ind_hc2_win_gfl       = 72   !< index in input list for heat capacity at second window layer,
                                                  !< ground floor level
    INTEGER(iwp) ::  ind_hc2_win_r         = 108  !< index in input list for heat capacity at second window layer, roof
    INTEGER(iwp) ::  ind_hc3_agfl          = 8    !< index in input list for heat capacity at third wall layer,
                                                  !< above ground floor level
    INTEGER(iwp) ::  ind_hc3_gfl           = 28   !< index in input list for heat capacity at third wall layer, ground floor level
    INTEGER(iwp) ::  ind_hc3_wall_r        = 96   !< index in input list for heat capacity at third wall layer, roof
    INTEGER(iwp) ::  ind_hc3_win_agfl      = 85   !< index in input list for heat capacity at third window layer,
                                                  !< above ground floor level
    INTEGER(iwp) ::  ind_hc3_win_gfl       = 73   !< index in input list for heat capacity at third window layer,
                                                  !< ground floor level
    INTEGER(iwp) ::  ind_hc3_win_r         = 109  !< index in input list for heat capacity at third window layer, roof
    INTEGER(iwp) ::  ind_hc4_agfl          = 136  !< index in input list for heat capacity at fourth wall layer, above ground floor level
    INTEGER(iwp) ::  ind_hc4_gfl           = 138  !< index in input list for heat capacity at fourth wall layer, ground floor level
    INTEGER(iwp) ::  ind_hc4_wall_r        = 146  !< index in input list for heat capacity at fourth wall layer, roof
    INTEGER(iwp) ::  ind_hc4_win_agfl      = 144  !< index in input list for heat capacity at fourth window layer,
                                                  !< above ground floor level
    INTEGER(iwp) ::  ind_hc4_win_gfl       = 142  !< index in input list for heat capacity at fourth window layer, ground floor level
    INTEGER(iwp) ::  ind_hc4_win_r         = 148  !< index in input list for heat capacity at fourth window layer, roof
    INTEGER(iwp) ::  ind_indoor_target_temp_summer = 12  !<
    INTEGER(iwp) ::  ind_indoor_target_temp_winter = 13  !<
    INTEGER(iwp) ::  ind_lai_r_agfl        = 4    !< index in input list for LAI on roof, above ground floor level
    INTEGER(iwp) ::  ind_lai_r_gfl         = 4    !< index in input list for LAI on roof, ground floor level
    INTEGER(iwp) ::  ind_lai_w_agfl        = 5    !< index in input list for LAI on wall, above ground floor level
    INTEGER(iwp) ::  ind_lai_w_gfl         = 25   !< index in input list for LAI on wall, ground floor level
    INTEGER(iwp) ::  ind_tc1_agfl          = 9    !< index in input list for thermal conductivity at first wall layer, above ground floor level
    INTEGER(iwp) ::  ind_tc1_gfl           = 29   !< index in input list for thermal conductivity at first wall layer,
                                                  !< ground floor level
    INTEGER(iwp) ::  ind_tc1_wall_r        = 97   !< index in input list for thermal conductivity at first wall layer, roof
    INTEGER(iwp) ::  ind_tc1_win_agfl      = 86   !< index in input list for thermal conductivity at first window layer,
                                                  !< above ground floor level
    INTEGER(iwp) ::  ind_tc1_win_gfl       = 74   !< index in input list for thermal conductivity at first window layer,
                                                  !< ground floor level
    INTEGER(iwp) ::  ind_tc1_win_r         = 110  !< index in input list for thermal conductivity at first window layer, roof
    INTEGER(iwp) ::  ind_tc2_agfl          = 10   !< index in input list for thermal conductivity at second wall layer,
                                                  !< above ground floor level
    INTEGER(iwp) ::  ind_tc2_gfl           = 30   !< index in input list for thermal conductivity at second wall layer,
                                                  !< ground floor level
    INTEGER(iwp) ::  ind_tc2_wall_r        = 98   !< index in input list for thermal conductivity at second wall layer, roof
    INTEGER(iwp) ::  ind_tc2_win_agfl      = 87   !< index in input list for thermal conductivity at second window layer,
                                                  !< above ground floor level
    INTEGER(iwp) ::  ind_tc2_win_gfl       = 75   !< index in input list for thermal conductivity at second window layer,
                                                  !< ground floor level
    INTEGER(iwp) ::  ind_tc2_win_r         = 111  !< index in input list for thermal conductivity at second window layer,
                                                  !< ground floor level
    INTEGER(iwp) ::  ind_tc3_agfl          = 11   !< index in input list for thermal conductivity at third wall layer,
                                                  !< above ground floor level
    INTEGER(iwp) ::  ind_tc3_gfl           = 31   !< index in input list for thermal conductivity at third wall layer,
                                                  !< ground floor level
    INTEGER(iwp) ::  ind_tc3_wall_r        = 99   !< index in input list for thermal conductivity at third wall layer, roof
    INTEGER(iwp) ::  ind_tc3_win_agfl      = 88   !< index in input list for thermal conductivity at third window layer,
                                                  !< above ground floor level
    INTEGER(iwp) ::  ind_tc3_win_gfl       = 76   !< index in input list for thermal conductivity at third window layer,
                                                  !< ground floor level
    INTEGER(iwp) ::  ind_tc3_win_r         = 112  !< index in input list for thermal conductivity at third window layer, roof
    INTEGER(iwp) ::  ind_tc4_agfl          = 137  !< index in input list for thermal conductivity at fourth wall layer,
                                                  !< above ground floor level
    INTEGER(iwp) ::  ind_tc4_gfl           = 139  !< index in input list for thermal conductivity at fourth wall layer,
                                                  !< ground floor level
    INTEGER(iwp) ::  ind_tc4_wall_r        = 147  !< index in input list for thermal conductivity at fourth wall layer, roof
    INTEGER(iwp) ::  ind_tc4_win_agfl      = 145  !< index in input list for thermal conductivity at fourth window layer,
                                                  !< above ground floor level
    INTEGER(iwp) ::  ind_tc4_win_gfl       = 143  !< index in input list for thermal conductivity at first window layer, ground floor level
    INTEGER(iwp) ::  ind_tc4_win_r         = 149  !< index in input list for thermal conductivity at third window layer, roof
    INTEGER(iwp) ::  ind_thick_1_agfl      = 41   !< index for wall layer thickness - 1st layer above ground floor level
    INTEGER(iwp) ::  ind_thick_1_gfl       = 62   !< index for wall layer thickness - 1st layer ground floor level
    INTEGER(iwp) ::  ind_thick_1_wall_r    = 90   !< index for wall layer thickness - 1st layer roof
    INTEGER(iwp) ::  ind_thick_1_win_agfl  = 79   !< index for window layer thickness - 1st layer above ground floor level
    INTEGER(iwp) ::  ind_thick_1_win_gfl   = 67   !< index for window layer thickness - 1st layer ground floor level
    INTEGER(iwp) ::  ind_thick_1_win_r     = 103  !< index for window layer thickness - 1st layer roof
    INTEGER(iwp) ::  ind_thick_2_agfl      = 42   !< index for wall layer thickness - 2nd layer above ground floor level
    INTEGER(iwp) ::  ind_thick_2_gfl       = 63   !< index for wall layer thickness - 2nd layer ground floor level
    INTEGER(iwp) ::  ind_thick_2_wall_r    = 91   !< index for wall layer thickness - 2nd layer roof
    INTEGER(iwp) ::  ind_thick_2_win_agfl  = 80   !< index for window layer thickness - 2nd layer above ground floor level
    INTEGER(iwp) ::  ind_thick_2_win_gfl   = 68   !< index for window layer thickness - 2nd layer ground floor level
    INTEGER(iwp) ::  ind_thick_2_win_r     = 104  !< index for window layer thickness - 2nd layer roof
    INTEGER(iwp) ::  ind_thick_3_agfl      = 43   !< index for wall layer thickness - 3rd layer above ground floor level
    INTEGER(iwp) ::  ind_thick_3_gfl       = 64   !< index for wall layer thickness - 3rd layer ground floor level
    INTEGER(iwp) ::  ind_thick_3_wall_r    = 92   !< index for wall layer thickness - 3rd layer roof
    INTEGER(iwp) ::  ind_thick_3_win_agfl  = 81   !< index for window layer thickness - 3rd layer above ground floor level
    INTEGER(iwp) ::  ind_thick_3_win_gfl   = 69   !< index for window layer thickness - 3rd layer ground floor level
    INTEGER(iwp) ::  ind_thick_3_win_r     = 105  !< index for window layer thickness - 3rd layer roof
    INTEGER(iwp) ::  ind_thick_4_agfl      = 44   !< index for wall layer thickness - 4th layer above ground floor level
    INTEGER(iwp) ::  ind_thick_4_gfl       = 65   !< index for wall layer thickness - 4th layer ground floor level
    INTEGER(iwp) ::  ind_thick_4_wall_r    = 93   !< index for wall layer thickness - 4st layer roof
    INTEGER(iwp) ::  ind_thick_4_win_agfl  = 82   !< index for window layer thickness - 4th layer above ground floor level
    INTEGER(iwp) ::  ind_thick_4_win_gfl   = 70   !< index for window layer thickness - 4th layer ground floor level
    INTEGER(iwp) ::  ind_thick_4_win_r     = 106  !< index for window layer thickness - 4th layer roof
    INTEGER(iwp) ::  ind_trans_agfl        = 17   !< index in input list for window transmissivity, above ground floor level
    INTEGER(iwp) ::  ind_trans_gfl         = 35   !< index in input list for window transmissivity, ground floor level
    INTEGER(iwp) ::  ind_trans_r           = 114  !< index in input list for window transmissivity, roof
    INTEGER(iwp) ::  ind_wall_frac_agfl    = 0    !< index in input list for wall fraction, above ground floor level
    INTEGER(iwp) ::  ind_wall_frac_gfl     = 21   !< index in input list for wall fraction, ground floor level
    INTEGER(iwp) ::  ind_wall_frac_r       = 89   !< index in input list for wall fraction, roof
    INTEGER(iwp) ::  ind_win_frac_agfl     = 1    !< index in input list for window fraction, above ground floor level
    INTEGER(iwp) ::  ind_win_frac_gfl      = 22   !< index in input list for window fraction, ground floor level
    INTEGER(iwp) ::  ind_win_frac_r        = 102  !< index in input list for window fraction, roof
    INTEGER(iwp) ::  ind_z0_agfl           = 18   !< index in input list for z0, above ground floor level
    INTEGER(iwp) ::  ind_z0_gfl            = 36   !< index in input list for z0, ground floor level
    INTEGER(iwp) ::  ind_z0qh_agfl         = 19   !< index in input list for z0h / z0q, above ground floor level
    INTEGER(iwp) ::  ind_z0qh_gfl          = 37   !< index in input list for z0h / z0q, ground floor level
!
!-- Indices of input attributes in building_surface_pars (except for radiation-related, which are in
!-- radiation_model_mod)
    CHARACTER(37), DIMENSION(0:7), PARAMETER ::  building_type_name = (/     &
                                   'user-defined                         ', &  !< type 0
                                   'residential - 1950                   ', &  !< type  1
                                   'residential 1951 - 2000              ', &  !< type  2
                                   'residential 2001 -                   ', &  !< type  3
                                   'office - 1950                        ', &  !< type  4
                                   'office 1951 - 2000                   ', &  !< type  5
                                   'office 2001 -                        ', &  !< type  6
                                   'bridges                              '  &  !< type  7
                                                                     /)

    INTEGER(iwp) ::  ind_s_emis_green                = 14  !< index for emissivity of green fraction (0-1)
    INTEGER(iwp) ::  ind_s_emis_wall                 = 13  !< index for emissivity of wall fraction (0-1)
    INTEGER(iwp) ::  ind_s_emis_win                  = 15  !< index for emissivity o f window fraction (0-1)
    INTEGER(iwp) ::  ind_s_green_frac_r              = 3   !< index for green fraction on roof (0-1)
    INTEGER(iwp) ::  ind_s_green_frac_w              = 2   !< index for green fraction on wall (0-1)
    INTEGER(iwp) ::  ind_s_hc1                       = 5   !< index for heat capacity of wall layer 1
    INTEGER(iwp) ::  ind_s_hc2                       = 6   !< index for heat capacity of wall layer 2
    INTEGER(iwp) ::  ind_s_hc3                       = 7   !< index for heat capacity of wall layer 3
    INTEGER(iwp) ::  ind_s_indoor_target_temp_summer = 11  !< index for indoor target summer temperature
    INTEGER(iwp) ::  ind_s_indoor_target_temp_winter = 12  !< index for indoor target winter temperature
    INTEGER(iwp) ::  ind_s_lai_r                     = 4   !< index for leaf area index of green fraction
    INTEGER(iwp) ::  ind_s_tc1                       = 8   !< index for thermal conducivity of wall layer 1
    INTEGER(iwp) ::  ind_s_tc2                       = 9   !< index for thermal conducivity of wall layer 2
    INTEGER(iwp) ::  ind_s_tc3                       = 10  !< index for thermal conducivity of wall layer 3
    INTEGER(iwp) ::  ind_s_trans                     = 16  !< index for transmissivity of window fraction (0-1)
    INTEGER(iwp) ::  ind_s_wall_frac                 = 0   !< index for wall fraction (0-1)
    INTEGER(iwp) ::  ind_s_win_frac                  = 1   !< index for window fraction (0-1)
    INTEGER(iwp) ::  ind_s_z0                        = 17  !< index for roughness length for momentum (m)
    INTEGER(iwp) ::  ind_s_z0qh                      = 18  !< index for roughness length for heat (m)

    REAL(wp) ::  dt_usm = HUGE( 1.0_wp )                   !< maximum allowed timestep of the urban-surface model
    REAL(wp) ::  ground_floor_level = 4.0_wp               !< default ground floor level

!
!-- Building facade/wall/green/window properties (partly according to PIDS).
!-- Initialization of building_pars is outsourced to usm_init_pars. This is needed because of the
!-- huge number of attributes given in building_pars (>700), while intel and gfortran compiler have
!-- hard limit of continuation lines of 511.
    REAL(wp), DIMENSION(0:149,1:7) ::  building_pars  !<
!
!-- Type for 1d surface variables as surface temperature and liquid water reservoir
    TYPE surf_type_1d_usm
       REAL(wp), DIMENSION(:), ALLOCATABLE ::  val  !<
    END TYPE surf_type_1d_usm
!
!-- Type for 2d surface variables as wall temperature
    TYPE surf_type_2d_usm
       REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  val  !<
    END TYPE surf_type_2d_usm
!-- Wall surface model
!-- Wall surface model constants
    INTEGER(iwp), PARAMETER ::  nzb_wall = 0  !< inner side of the wall model (to be switched)
    INTEGER(iwp), PARAMETER ::  nzt_wall = 3  !< outer side of the wall model (to be switched)
    INTEGER(iwp), PARAMETER ::  nzw      = 4  !< number of wall layers (fixed for now)

    INTEGER(iwp)            ::  soil_type     !<

    REAL(wp)  ::  m_total                  = 0.0_wp    !< weighted total water content of the soil (m3/m3)
    REAL(wp)  ::  roof_inner_temperature   = 295.0_wp  !< temperature of the inner roof
                                                       !< surface (~22 degrees C) (K)
    REAL(wp)  ::  wall_inner_temperature   = 295.0_wp  !< temperature of the inner wall
                                                       !< surface (~22 degrees C) (K)
    REAL(wp)  ::  window_inner_temperature = 295.0_wp  !< temperature of the inner window
                                                       !< surface (~22 degrees C) (K)
!
!-- Surface and material model variables for walls, ground, roofs
    TYPE(surf_type_1d_usm), POINTER ::  t_surf_green      !< prognostic array for green surface temperature
    TYPE(surf_type_1d_usm), POINTER ::  t_surf_green_p    !< prognostic array for green surface temperature
    TYPE(surf_type_1d_usm), POINTER ::  t_surf_wall       !< prognostic array for wall surface temperature
    TYPE(surf_type_1d_usm), POINTER ::  t_surf_wall_p     !< prognostic array for wall surface temperature
    TYPE(surf_type_1d_usm), POINTER ::  t_surf_window     !< prognostic array for window surface temperature
    TYPE(surf_type_1d_usm), POINTER ::  t_surf_window_p   !< prognostic array for window surface temperature

    TYPE(surf_type_1d_usm), TARGET ::  t_surf_green_1    !<
    TYPE(surf_type_1d_usm), TARGET ::  t_surf_green_2    !<
    TYPE(surf_type_1d_usm), TARGET ::  t_surf_wall_1     !<
    TYPE(surf_type_1d_usm), TARGET ::  t_surf_wall_2     !<
    TYPE(surf_type_1d_usm), TARGET ::  t_surf_window_1   !<
    TYPE(surf_type_1d_usm), TARGET ::  t_surf_window_2   !<

!
!-- Energy balance variables
!-- Parameters of the land, roof and wall surfaces (only for horizontally upward surfaces)
    TYPE(surf_type_1d_usm), POINTER ::  m_liq_usm   !< liquid water reservoir (m), horizontal surface elements
    TYPE(surf_type_1d_usm), POINTER ::  m_liq_usm_p !< progn. liquid water reservoir (m), horizontal surface elements

    TYPE(surf_type_1d_usm), TARGET ::  m_liq_usm_1  !<
    TYPE(surf_type_1d_usm), TARGET ::  m_liq_usm_2  !<
    TYPE(surf_type_1d_usm), TARGET ::  tm_liq_usm_m !< liquid water reservoir tendency (m), horizontal surface elements

    TYPE(surf_type_2d_usm), POINTER ::  fc          !<
    TYPE(surf_type_2d_usm), POINTER ::  rootfr      !<
    TYPE(surf_type_2d_usm), POINTER ::  swc         !<
    TYPE(surf_type_2d_usm), POINTER ::  swc_p       !<
    TYPE(surf_type_2d_usm), POINTER ::  swc_res     !<
    TYPE(surf_type_2d_usm), POINTER ::  swc_sat     !<
    TYPE(surf_type_2d_usm), POINTER ::  t_green     !<
    TYPE(surf_type_2d_usm), POINTER ::  t_green_p   !<
    TYPE(surf_type_2d_usm), POINTER ::  t_wall      !<
    TYPE(surf_type_2d_usm), POINTER ::  t_wall_p    !<
    TYPE(surf_type_2d_usm), POINTER ::  wilt        !<
    TYPE(surf_type_2d_usm), POINTER ::  t_window    !<
    TYPE(surf_type_2d_usm), POINTER ::  t_window_p  !<


    TYPE(surf_type_2d_usm), TARGET ::  fc_1        !<
    TYPE(surf_type_2d_usm), TARGET ::  rootfr_1    !<
    TYPE(surf_type_2d_usm), TARGET ::  swc_1       !<
    TYPE(surf_type_2d_usm), TARGET ::  swc_2       !<
    TYPE(surf_type_2d_usm), TARGET ::  swc_res_1   !<
    TYPE(surf_type_2d_usm), TARGET ::  swc_sat_1   !<
    TYPE(surf_type_2d_usm), TARGET ::  t_green_1   !<
    TYPE(surf_type_2d_usm), TARGET ::  t_green_2   !<
    TYPE(surf_type_2d_usm), TARGET ::  t_wall_1    !<
    TYPE(surf_type_2d_usm), TARGET ::  t_wall_2    !<
    TYPE(surf_type_2d_usm), TARGET ::  wilt_1      !<
    TYPE(surf_type_2d_usm), TARGET ::  t_window_1  !<
    TYPE(surf_type_2d_usm), TARGET ::  t_window_2  !<

!
!-- Arrays for time averages
    TYPE(surf_type_1d_usm) ::  wghf_eb_av          !< average of wghf_eb
    TYPE(surf_type_1d_usm) ::  wghf_eb_window_av   !< average of wghf_eb window
    TYPE(surf_type_1d_usm) ::  wghf_eb_green_av    !< average of wghf_eb window
    TYPE(surf_type_1d_usm) ::  iwghf_eb_av         !< indoor average of wghf_eb
    TYPE(surf_type_1d_usm) ::  iwghf_eb_window_av  !< indoor average of wghf_eb window
    TYPE(surf_type_1d_usm) ::  wshf_eb_av          !< average of wshf_eb
    TYPE(surf_type_1d_usm) ::  qsws_av             !< average of qsws
    TYPE(surf_type_1d_usm) ::  qsws_veg_av         !< average of qsws_veg_eb
    TYPE(surf_type_1d_usm) ::  qsws_liq_av         !< average of qsws_liq_eb
    TYPE(surf_type_1d_usm) ::  t_surf_wall_av      !< average of wall surface temperature (K)
    TYPE(surf_type_1d_usm) ::  t_surf_window_av    !< average of window surface temperature (K)
    TYPE(surf_type_1d_usm) ::  t_surf_green_av     !< average of green wall surface temperature (K)

    TYPE(surf_type_2d_usm) ::  t_wall_av           !< average of t_wall
    TYPE(surf_type_2d_usm) ::  t_window_av         !< average of t_window
    TYPE(surf_type_2d_usm) ::  t_green_av          !< average of t_green
    TYPE(surf_type_2d_usm) ::  swc_av              !< average of swc

!
!-- Interfaces of subroutines accessed from outside of this module
    INTERFACE usm_3d_data_averaging
       MODULE PROCEDURE usm_3d_data_averaging
    END INTERFACE usm_3d_data_averaging

    INTERFACE usm_boundary_condition
       MODULE PROCEDURE usm_boundary_condition
    END INTERFACE usm_boundary_condition

    INTERFACE usm_check_data_output
       MODULE PROCEDURE usm_check_data_output
    END INTERFACE usm_check_data_output

    INTERFACE usm_check_parameters
       MODULE PROCEDURE usm_check_parameters
    END INTERFACE usm_check_parameters

    INTERFACE usm_data_output_3d
       MODULE PROCEDURE usm_data_output_3d
    END INTERFACE usm_data_output_3d

    INTERFACE usm_define_netcdf_grid
       MODULE PROCEDURE usm_define_netcdf_grid
    END INTERFACE usm_define_netcdf_grid

    INTERFACE usm_init
       MODULE PROCEDURE usm_init
    END INTERFACE usm_init

    INTERFACE usm_init_arrays
       MODULE PROCEDURE usm_init_arrays
    END INTERFACE usm_init_arrays

    INTERFACE usm_parin
       MODULE PROCEDURE usm_parin
    END INTERFACE usm_parin

    INTERFACE usm_rrd_local
       MODULE PROCEDURE usm_rrd_local_ftn
       MODULE PROCEDURE usm_rrd_local_mpi
    END INTERFACE usm_rrd_local

    INTERFACE usm_energy_balance
       MODULE PROCEDURE usm_energy_balance
    END INTERFACE usm_energy_balance

    INTERFACE usm_swap_timelevel
       MODULE PROCEDURE usm_swap_timelevel
    END INTERFACE usm_swap_timelevel

    INTERFACE usm_timestep
       MODULE PROCEDURE usm_timestep
    END INTERFACE usm_timestep

    INTERFACE usm_vm_sampling
       MODULE PROCEDURE usm_vm_sampling
    END INTERFACE usm_vm_sampling

    INTERFACE usm_wrd_local
       MODULE PROCEDURE usm_wrd_local
    END INTERFACE usm_wrd_local


    SAVE

    PRIVATE

!
!-- Public functions
    PUBLIC usm_boundary_condition,                                                                 &
           usm_check_data_output,                                                                  &
           usm_check_parameters,                                                                   &
           usm_data_output_3d,                                                                     &
           usm_define_netcdf_grid,                                                                 &
           usm_init,                                                                               &
           usm_init_arrays,                                                                        &
           usm_parin,                                                                              &
           usm_rrd_local,                                                                          &
           usm_energy_balance,                                                                     &
           usm_swap_timelevel,                                                                     &
           usm_timestep,                                                                           &
           usm_vm_sampling,                                                                        &
           usm_wrd_local,                                                                          &
           usm_3d_data_averaging

!
!-- Public parameters, constants and initial values
    PUBLIC building_type,                                                                          &
           building_pars,                                                                          &
           dt_usm,                                                                                 &
           nzb_wall,                                                                               &
           nzt_wall,                                                                               &
           t_green,                                                                                &
           t_wall,                                                                                 &
           t_window,                                                                               &
           usm_wall_mod

 CONTAINS

!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> This subroutine creates the necessary indices of the urban surfaces and plant canopy and it
!> allocates the needed arrays for USM
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE usm_init_arrays

    IMPLICIT NONE

    IF ( debug_output )  CALL debug_message( 'usm_init_arrays', 'start' )

!
!-- Allocate radiation arrays which are part of the new data type.
!-- For horizontal surfaces.
    ALLOCATE ( surf_usm%surfhf(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%rad_net_l(1:surf_usm%ns) )

!
!-- Wall surface model
!-- Allocate arrays for wall surface model and define pointers
!-- Allocate array of wall types and wall parameters

    ALLOCATE ( surf_usm%surface_types(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%building_type(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%building_type_name(1:surf_usm%ns) )
    surf_usm%building_type      = 0
    surf_usm%building_type_name = 'none'
!
!-- Allocate array to store maximum allowed timestep on each surface element. Later on,
!-- this will be used to limit the model time step accordingly. Note, in most of the cases
!-- the allowed wall timestep is much larger than the timestep allowed for the 3D flow model.
    ALLOCATE( surf_usm%dt_max(1:surf_usm%ns) )

!
!-- Allocate albedo_type and albedo. Each surface element has 3 values, 0: wall fraction,
!-- 1: green fraction, 2: window fraction.
    ALLOCATE ( surf_usm%albedo_type(1:surf_usm%ns,0:2) )
    ALLOCATE ( surf_usm%albedo(1:surf_usm%ns,0:2) )
    surf_usm%albedo_type = albedo_type
!
!-- Allocate indoor target temperature for summer and winter
    ALLOCATE ( surf_usm%target_temp_summer(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%target_temp_winter(1:surf_usm%ns) )
!
!-- Allocate flag indicating ground floor level surface elements
    ALLOCATE ( surf_usm%gfl(1:surf_usm%ns) )
!
!-- Allocate arrays for relative surface fraction.
!-- 0 - wall fraction, 1 - green fraction, 2 - window fraction
    ALLOCATE ( surf_usm%frac(1:surf_usm%ns,0:2) )
    surf_usm%frac = 0.0_wp
!
!-- Wall and roof surface parameters.
    ALLOCATE ( surf_usm%isroof_surf(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%lambda_surf(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%lambda_surf_window(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%lambda_surf_green(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%c_surface(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%c_surface_window(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%c_surface_green(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%transmissivity(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%lai(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%emissivity(1:surf_usm%ns,0:2) )
    ALLOCATE ( surf_usm%r_a(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%r_a_green(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%r_a_window(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%green_type_roof(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%r_s(1:surf_usm%ns) )
!
!-- Allocate wall and roof material parameters.
    ALLOCATE ( surf_usm%thickness_wall(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%thickness_window(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%thickness_green(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%lambda_h(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%lambda_h_layer(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%rho_c_wall(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%lambda_h_window(nzb_wall:nzt_wall,1:surf_usm%ns)  )
    ALLOCATE ( surf_usm%lambda_h_window_layer(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%rho_c_window(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%lambda_h_green(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%rho_c_green(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%rho_c_total_green(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%n_vg_green(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%alpha_vg_green(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%l_vg_green(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%gamma_w_green_sat(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%lambda_w_green(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%lambda_w_green_layer(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%gamma_w_green(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%gamma_w_green_layer(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%tswc_m(nzb_wall:nzt_wall,1:surf_usm%ns) )


!
!-- Allocate green wall and roof vegetation and soil parameters.
    ALLOCATE ( surf_usm%g_d(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%c_liq(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%qsws_liq(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%qsws_veg(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%r_canopy(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%r_canopy_min(1:surf_usm%ns) )
!
!-- Allocate wall and roof layers sizes.
    ALLOCATE ( surf_usm%dz_wall(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%dz_window(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%dz_green(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%ddz_wall(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%dz_wall_center(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%ddz_wall_center(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%zw(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%ddz_window(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%dz_window_center(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%ddz_window_center(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%zw_window(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%ddz_green(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%dz_green_center(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%ddz_green_center(nzb_wall:nzt_wall,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%zw_green(nzb_wall:nzt_wall,1:surf_usm%ns) )
!
!-- Allocate wall and roof temperature arrays.
!-- Allocate if required. Note, in case of restarts, some of these arrays might be already allocated.
    IF ( .NOT. ALLOCATED( t_surf_wall_1%val ) )    ALLOCATE ( t_surf_wall_1%val(1:surf_usm%ns) )
    IF ( .NOT. ALLOCATED( t_surf_wall_2%val ) )    ALLOCATE ( t_surf_wall_2%val(1:surf_usm%ns) )
    IF ( .NOT. ALLOCATED( t_wall_1%val ) )         ALLOCATE ( t_wall_1%val(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
    IF ( .NOT. ALLOCATED( t_wall_2%val ) )         ALLOCATE ( t_wall_2%val(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
    IF ( .NOT. ALLOCATED( t_surf_window_1%val ) )  ALLOCATE ( t_surf_window_1%val(1:surf_usm%ns) )
    IF ( .NOT. ALLOCATED( t_surf_window_2%val ) )  ALLOCATE ( t_surf_window_2%val(1:surf_usm%ns) )
    IF ( .NOT. ALLOCATED( t_window_1%val ) )       ALLOCATE ( t_window_1%val(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
    IF ( .NOT. ALLOCATED( t_window_2%val ) )       ALLOCATE ( t_window_2%val(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
    IF ( .NOT. ALLOCATED( t_surf_green_1%val ) )   ALLOCATE ( t_surf_green_1%val(1:surf_usm%ns) )
    IF ( .NOT. ALLOCATED( t_surf_green_2%val ) )   ALLOCATE ( t_surf_green_2%val(1:surf_usm%ns) )
    IF ( .NOT. ALLOCATED( t_green_1%val ) )        ALLOCATE ( t_green_1%val(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
    IF ( .NOT. ALLOCATED( t_green_2%val ) )        ALLOCATE ( t_green_2%val(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
    IF ( .NOT. ALLOCATED( swc_1%val ) )            ALLOCATE ( swc_1%val(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
    IF ( .NOT. ALLOCATED( swc_sat_1%val ) )        ALLOCATE ( swc_sat_1%val(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
    IF ( .NOT. ALLOCATED( swc_res_1%val ) )        ALLOCATE ( swc_res_1%val(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
    IF ( .NOT. ALLOCATED( swc_2%val ) )            ALLOCATE ( swc_2%val(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
    IF ( .NOT. ALLOCATED( rootfr_1%val ) )         ALLOCATE ( rootfr_1%val(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
    IF ( .NOT. ALLOCATED( wilt_1%val ) )           ALLOCATE ( wilt_1%val(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
    IF ( .NOT. ALLOCATED( fc_1%val ) )             ALLOCATE ( fc_1%val(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
    IF ( .NOT. ALLOCATED( m_liq_usm_1%val ) )      ALLOCATE ( m_liq_usm_1%val(1:surf_usm%ns) )
    IF ( .NOT. ALLOCATED( m_liq_usm_2%val ) )      ALLOCATE ( m_liq_usm_2%val(1:surf_usm%ns) )
!
!-- Initial assignment of the pointers
    t_wall    => t_wall_1;   t_wall_p   => t_wall_2
    t_window  => t_window_1; t_window_p => t_window_2
    t_green   => t_green_1;  t_green_p  => t_green_2
    t_surf_wall   => t_surf_wall_1;   t_surf_wall_p   => t_surf_wall_2
    t_surf_window => t_surf_window_1; t_surf_window_p => t_surf_window_2
    t_surf_green  => t_surf_green_1;  t_surf_green_p  => t_surf_green_2
    m_liq_usm     => m_liq_usm_1;     m_liq_usm_p     => m_liq_usm_2
    swc     => swc_1; swc_p => swc_2
    swc_sat => swc_sat_1
    swc_res => swc_res_1
    rootfr  => rootfr_1
    wilt    => wilt_1
    fc      => fc_1

!
!-- Allocate intermediate timestep arrays. For horizontal surfaces.
    ALLOCATE ( surf_usm%tt_surface_wall_m(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%tt_wall_m(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%tt_surface_window_m(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%tt_window_m(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%tt_green_m(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
    ALLOCATE ( surf_usm%tt_surface_green_m(1:surf_usm%ns) )
!
!-- Allocate intermediate timestep arrays
    ALLOCATE ( tm_liq_usm_m%val(1:surf_usm%ns) )
    tm_liq_usm_m%val = 0.0_wp
!
!-- Set inital values for prognostic quantities
    IF ( ALLOCATED( surf_usm%tt_surface_wall_m )   )  surf_usm%tt_surface_wall_m   = 0.0_wp
    IF ( ALLOCATED( surf_usm%tt_wall_m )           )  surf_usm%tt_wall_m           = 0.0_wp
    IF ( ALLOCATED( surf_usm%tt_surface_window_m ) )  surf_usm%tt_surface_window_m = 0.0_wp
    IF ( ALLOCATED( surf_usm%tt_window_m    )      )  surf_usm%tt_window_m         = 0.0_wp
    IF ( ALLOCATED( surf_usm%tt_green_m    )       )  surf_usm%tt_green_m          = 0.0_wp
    IF ( ALLOCATED( surf_usm%tt_surface_green_m )  )  surf_usm%tt_surface_green_m  = 0.0_wp
!
!-- Allocate wall heat flux output arrays and set initial values.
    ALLOCATE ( surf_usm%ghf(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%wshf_eb(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%wghf_eb(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%wghf_eb_window(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%wghf_eb_green(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%iwghf_eb(1:surf_usm%ns) )
    ALLOCATE ( surf_usm%iwghf_eb_window(1:surf_usm%ns) )
    IF ( ALLOCATED( surf_usm%ghf     ) )  surf_usm%ghf     = 0.0_wp
    IF ( ALLOCATED( surf_usm%wshf_eb ) )  surf_usm%wshf_eb = 0.0_wp
    IF ( ALLOCATED( surf_usm%wghf_eb ) )  surf_usm%wghf_eb = 0.0_wp
    IF ( ALLOCATED( surf_usm%wghf_eb_window ) )   surf_usm%wghf_eb_window  = 0.0_wp
    IF ( ALLOCATED( surf_usm%wghf_eb_green ) )    surf_usm%wghf_eb_green   = 0.0_wp
    IF ( ALLOCATED( surf_usm%iwghf_eb ) )         surf_usm%iwghf_eb        = 0.0_wp
    IF ( ALLOCATED( surf_usm%iwghf_eb_window ) )  surf_usm%iwghf_eb_window = 0.0_wp
!
!-- Initialize building-surface properties, which are also required by other modules, e.g. the
!-- indoor model.
    CALL usm_define_pars

    IF ( debug_output )  CALL debug_message( 'usm_init_arrays', 'end' )

 END SUBROUTINE usm_init_arrays


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Sum up and time-average urban surface output quantities as well as allocate the array necessary
!> for storing the average.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE usm_3d_data_averaging( mode, variable )

    IMPLICIT NONE

    CHARACTER(LEN=*), INTENT(IN) ::  variable  !<
    CHARACTER(LEN=*), INTENT(IN) ::  mode      !<
    CHARACTER(LEN=varnamelength) ::  var       !< trimmed variable

    INTEGER(iwp) ::  i, j, k, m, ids, idsint, iwl, istat  !< running indices

    LOGICAL ::  downward  !< control flag indicating the output of downward-facing surfaces
    LOGICAL ::  eastward  !< control flag indicating the output of east-facing surfaces
    LOGICAL ::  northward !< control flag indicating the output of northward-facing surfaces
    LOGICAL ::  southward !< control flag indicating the output of southward-facing surfaces
    LOGICAL ::  upward    !< control flag indicating the output of upward-facing surfaces
    LOGICAL ::  westward  !< control flag indicating the output of westward-facing surfaces


    IF ( .NOT. variable(1:4) == 'usm_' )  RETURN  ! Is such a check really required?

!
!-- Find the real name of the variable
    ids = -1
    var = TRIM( variable )
    DO  i = 0, nd-1
       k = len( TRIM( var ) )
       j = len( TRIM( dirname(i) ) )
       IF ( TRIM( var(k-j+1:k) ) == TRIM( dirname(i) ) )  THEN
           ids = i
           idsint = dirint(ids)
           var = var(:k-j)
           EXIT
       ENDIF
    ENDDO
    IF ( ids == -1 )  THEN
        var = TRIM( variable )
    ELSE
!
!--    Set direction control flags
       downward  = .FALSE.
       eastward  = .FALSE.
       northward = .FALSE.
       southward = .FALSE.
       upward    = .FALSE.
       westward  = .FALSE.
       IF ( idsint == iup )  THEN
          upward = .TRUE.
       ELSEIF ( idsint == idown )  THEN
          downward = .TRUE.
       ELSEIF ( idsint == ieast )  THEN
          eastward = .TRUE.
       ELSEIF ( idsint == iwest )  THEN
          westward = .TRUE.
       ELSEIF ( idsint == inorth )  THEN
          northward = .TRUE.
       ELSEIF ( idsint == isouth )  THEN
          southward = .TRUE.
       ENDIF
    ENDIF
    IF ( var(1:11) == 'usm_t_wall_'  .AND.  len( TRIM( var ) ) >= 12 )  THEN
!
!--    Wall layers
       READ( var(12:12), '(I1)', iostat=istat ) iwl
       IF ( istat == 0  .AND.  iwl >= nzb_wall  .AND.  iwl <= nzt_wall )  THEN
          var = var(1:10)
       ELSE
!
!--       Wrong wall layer index
          RETURN
       ENDIF
    ENDIF
    IF ( var(1:13) == 'usm_t_window_'  .AND.  len( TRIM(var) ) >= 14 )  THEN
!
!--      Wall layers
        READ( var(14:14), '(I1)', iostat=istat ) iwl
        IF ( istat == 0  .AND.  iwl >= nzb_wall  .AND.  iwl <= nzt_wall )  THEN
            var = var(1:12)
        ELSE
!
!--         Wrong window layer index
            RETURN
        ENDIF
    ENDIF
    IF ( var(1:12) == 'usm_t_green_'  .AND.  len( TRIM( var ) ) >= 13 )  THEN
!
!--      Wall layers
        READ( var(13:13), '(I1)', iostat=istat ) iwl
        IF ( istat == 0  .AND.  iwl >= nzb_wall  .AND.  iwl <= nzt_wall )  THEN
            var = var(1:11)
        ELSE
!
!--         Wrong green layer index
            RETURN
        ENDIF
    ENDIF
    IF ( var(1:8) == 'usm_swc_'  .AND.  len( TRIM( var ) ) >= 9 )  THEN
!
!--      Swc layers
        READ( var(9:9), '(I1)', iostat=istat ) iwl
        IF ( istat == 0  .AND.  iwl >= nzb_wall  .AND.  iwl <= nzt_wall )  THEN
            var = var(1:7)
        ELSE
!
!--         Wrong swc layer index
            RETURN
        ENDIF
    ENDIF

    IF ( mode == 'allocate' )  THEN

       SELECT CASE ( TRIM( var ) )

          CASE ( 'usm_wshf' )
!
!--          Sensible heat flux
             IF ( .NOT.  ALLOCATED( wshf_eb_av%val ) )  THEN
                ALLOCATE ( wshf_eb_av%val(1:surf_usm%ns) )
                wshf_eb_av%val = 0.0_wp
             ENDIF

          CASE ( 'usm_qsws' )
!
!--          Latent heat flux
             IF ( .NOT.  ALLOCATED( qsws_av%val ) )  THEN
                ALLOCATE ( qsws_av%val(1:surf_usm%ns) )
                qsws_av%val = 0.0_wp
             ENDIF

          CASE ( 'usm_qsws_veg' )
!
!--          Latent heat flux from vegetation surfaces
             IF ( .NOT.  ALLOCATED( qsws_veg_av%val ) )  THEN
                ALLOCATE ( qsws_veg_av%val(1:surf_usm%ns) )
                qsws_veg_av%val = 0.0_wp
             ENDIF

          CASE ( 'usm_qsws_liq' )
!
!--          Latent heat flux from surfaces with liquid
             IF ( .NOT.  ALLOCATED( qsws_liq_av%val ) )  THEN
                ALLOCATE ( qsws_liq_av%val(1:surf_usm%ns) )
                qsws_liq_av%val = 0.0_wp
             ENDIF
!
!--       Please note, the following output quantities belongs to the individual tile fractions -
!--       ground heat flux at wall-, window-, and green fraction. Aggregated ground-heat flux is
!--       treated accordingly in average_3d_data, sum_up_3d_data, etc..
          CASE ( 'usm_wghf' )
!
!--          Heat flux from ground (wall)
             IF ( .NOT.  ALLOCATED( wghf_eb_av%val ) )  THEN
                ALLOCATE ( wghf_eb_av%val(1:surf_usm%ns) )
                wghf_eb_av%val = 0.0_wp
             ENDIF

          CASE ( 'usm_wghf_window' )
!
!--          Heat flux from window ground
             IF ( .NOT.  ALLOCATED( wghf_eb_window_av%val ) )  THEN
                ALLOCATE ( wghf_eb_window_av%val(1:surf_usm%ns) )
                wghf_eb_window_av%val = 0.0_wp
             ENDIF

          CASE ( 'usm_wghf_green' )
!
!--          Heat flux from green ground
             IF ( .NOT.  ALLOCATED( wghf_eb_green_av%val ) )  THEN
                ALLOCATE ( wghf_eb_green_av%val(1:surf_usm%ns) )
                wghf_eb_green_av%val = 0.0_wp
             ENDIF

          CASE ( 'usm_iwghf' )
!
!--          Heat flux from indoor ground
             IF ( .NOT.  ALLOCATED( iwghf_eb_av%val ) )  THEN
                ALLOCATE ( iwghf_eb_av%val(1:surf_usm%ns) )
                iwghf_eb_av%val = 0.0_wp
             ENDIF

          CASE ( 'usm_iwghf_window' )
!
!--          Heat flux from indoor window ground
             IF ( .NOT.  ALLOCATED( iwghf_eb_window_av%val ) )  THEN
                ALLOCATE ( iwghf_eb_window_av%val(1:surf_usm%ns) )
                iwghf_eb_window_av%val = 0.0_wp
             ENDIF

          CASE ( 'usm_t_surf_wall' )
!
!--          Surface temperature for wall surfaces
             IF ( .NOT.  ALLOCATED( t_surf_wall_av%val ) )  THEN
                ALLOCATE ( t_surf_wall_av%val(1:surf_usm%ns) )
                t_surf_wall_av%val = 0.0_wp
             ENDIF

          CASE ( 'usm_t_surf_window' )
!
!--          Surface temperature for window surfaces
             IF ( .NOT.  ALLOCATED( t_surf_window_av%val ) )  THEN
                ALLOCATE ( t_surf_window_av%val(1:surf_usm%ns) )
                t_surf_window_av%val = 0.0_wp
             ENDIF

          CASE ( 'usm_t_surf_green' )
!
!--          Surface temperature for green surfaces
             IF ( .NOT.  ALLOCATED( t_surf_green_av%val ) )  THEN
                ALLOCATE ( t_surf_green_av%val(1:surf_usm%ns) )
                t_surf_green_av%val = 0.0_wp
             ENDIF

          CASE ( 'usm_t_wall' )
!
!--          Wall temperature for iwl layer
             IF ( .NOT.  ALLOCATED( t_wall_av%val ) )  THEN
                ALLOCATE ( t_wall_av%val(nzb_wall:nzt_wall,1:surf_usm%ns) )
                t_wall_av%val = 0.0_wp
             ENDIF

          CASE ( 'usm_t_window' )
!
!--          Window temperature for iwl layer
             IF ( .NOT.  ALLOCATED( t_window_av%val ) )  THEN
                ALLOCATE ( t_window_av%val(nzb_wall:nzt_wall,1:surf_usm%ns) )
                t_window_av%val = 0.0_wp
             ENDIF

          CASE ( 'usm_t_green' )
!
!--          Green temperature for iwl layer
             IF ( .NOT.  ALLOCATED( t_green_av%val ) )  THEN
                ALLOCATE ( t_green_av%val(nzb_wall:nzt_wall,1:surf_usm%ns) )
                t_green_av%val = 0.0_wp
             ENDIF

          CASE ( 'usm_swc' )
!
!--          Soil water content for iwl layer
             IF ( .NOT.  ALLOCATED( swc_av%val ) )  THEN
                ALLOCATE ( swc_av%val(nzb_wall:nzt_wall,1:surf_usm%ns) )
                swc_av%val = 0.0_wp
             ENDIF

          CASE DEFAULT
             CONTINUE

       END SELECT

    ELSEIF ( mode == 'sum' )  THEN

       SELECT CASE ( TRIM( var ) )

          CASE ( 'usm_wshf' )
!
!--          Sensible heat flux
             CALL average_surfaces( wshf_eb_av%val, surf_usm%wshf_eb )

          CASE ( 'usm_qsws' )
!
!--          Latent heat flux
             CALL average_surfaces( qsws_av%val, surf_usm%qsws )

          CASE ( 'usm_qsws_veg' )
!
!--          Latent heat flux from vegetation surfaces
             CALL average_surfaces( qsws_veg_av%val, surf_usm%qsws_veg )

          CASE ( 'usm_qsws_liq' )
!
!--          Latent heat flux from surfaces with liquid
             CALL average_surfaces( qsws_liq_av%val, surf_usm%qsws_liq )

          CASE ( 'usm_wghf' )
!
!--           Heat flux from ground (wall)
              CALL average_surfaces( wghf_eb_av%val, surf_usm%wghf_eb )

          CASE ( 'usm_wghf_window' )
!
!--          Heat flux from window ground
             CALL average_surfaces( wghf_eb_window_av%val, surf_usm%wghf_eb_window )

          CASE ( 'usm_wghf_green' )
!
!--           Heat flux from green ground
              CALL average_surfaces( wghf_eb_green_av%val, surf_usm%wghf_eb_green )

          CASE ( 'usm_iwghf' )
!
!--          Heat flux from indoor ground
             CALL average_surfaces( iwghf_eb_av%val, surf_usm%iwghf_eb )

          CASE ( 'usm_iwghf_window' )
!
!--          Heat flux from indoor window ground
             CALL average_surfaces( iwghf_eb_window_av%val, surf_usm%iwghf_eb_window )

          CASE ( 'usm_t_surf_wall' )
!
!--          Surface temperature of wall surfaces
             CALL average_surfaces( t_surf_wall_av%val, t_surf_wall%val )

          CASE ( 'usm_t_surf_window' )
!
!--          Surface temperature for window surfaces
             CALL average_surfaces( t_surf_window_av%val, t_surf_window%val )

          CASE ( 'usm_t_surf_green' )
!
!--           Surface temperature for green surfaces
              CALL average_surfaces( t_surf_green_av%val, t_surf_green%val )

          CASE ( 'usm_t_wall' )
!
!--          Wall temperature for iwl layer
             CALL average_surfaces( t_wall_av%val(iwl,:), t_wall%val(iwl,:) )

          CASE ( 'usm_t_window' )
!
!--          Window temperature for iwl layer
             CALL average_surfaces( t_window_av%val(iwl,:), t_window%val(iwl,:) )

          CASE ( 'usm_t_green' )
!
!--          Green temperature for iwl layer
             CALL average_surfaces( t_green_av%val(iwl,:), t_green%val(iwl,:) )

          CASE ( 'usm_swc' )
!
!--          Soil water content for iwl layer
             CALL average_surfaces( swc_av%val(iwl,:), swc%val(iwl,:) )

          CASE DEFAULT
             CONTINUE

       END SELECT

    ELSEIF ( mode == 'average' )  THEN

       SELECT CASE ( TRIM( var ) )

          CASE ( 'usm_wshf' )
!
!--          Sensible heat flux
             CALL average_surfaces( wshf_eb_av%val )

          CASE ( 'usm_qsws' )
!
!--          Latent heat flux
             CALL average_surfaces( qsws_av%val )

          CASE ( 'usm_qsws_veg' )
!
!--          Latent heat flux from vegetation surfaces
             CALL average_surfaces( qsws_veg_av%val )

          CASE ( 'usm_qsws_liq' )
!
!--          Latent heat flux from surfaces with liquid
             CALL average_surfaces( qsws_liq_av%val )

          CASE ( 'usm_wghf' )
!
!--          Heat flux from ground
             CALL average_surfaces( wghf_eb_av%val )

          CASE ( 'usm_wghf_window' )
!
!--          Heat flux from window ground
             CALL average_surfaces( wghf_eb_window_av%val )

          CASE ( 'usm_wghf_green' )
!
!--          Heat flux from green ground
             CALL average_surfaces( wghf_eb_green_av%val )

          CASE ( 'usm_iwghf' )
!
!--          Heat flux from indoor ground
             CALL average_surfaces( iwghf_eb_av%val )

          CASE ( 'usm_iwghf_window' )
!
!--          Heat flux from indoor window ground
             CALL average_surfaces( iwghf_eb_window_av%val )

          CASE ( 'usm_t_surf_wall' )
!
!--          Surface temperature for wall surfaces
             CALL average_surfaces( t_surf_wall_av%val )

          CASE ( 'usm_t_surf_window' )
!
!--          Surface temperature for window surfaces
             CALL average_surfaces( t_surf_window_av%val )

          CASE ( 'usm_t_surf_green' )
!
!--          Surface temperature for green surfaces
             CALL average_surfaces( t_surf_green_av%val )

          CASE ( 'usm_t_wall' )
!
!--          Wall temperature for iwl layer
             CALL average_surfaces( t_wall_av%val(iwl,:) )

          CASE ( 'usm_t_window' )
!
!--          Window temperature for iwl layer
             CALL average_surfaces( t_window_av%val(iwl,:) )

          CASE ( 'usm_t_green' )
!
!--          Green temperature for iwl layer
             CALL average_surfaces( t_green_av%val(iwl,:) )

          CASE ( 'usm_swc' )
!
!--          Soil water content for iwl layer
             CALL average_surfaces( swc_av%val(iwl,:) )

       END SELECT

    ENDIF

    CONTAINS

!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Average surface data accorrding to its facing.
!--------------------------------------------------------------------------------------------------!
    SUBROUTINE average_surfaces( mean_array, input_array )

       REAL(wp), DIMENSION(1:surf_usm%ns), OPTIONAL ::  input_array !< array to be averaged and summed-up
       REAL(wp), DIMENSION(1:surf_usm%ns)           ::  mean_array  !< averaged and summed-up array


       IF ( mode == 'sum' )  THEN
!
!--       Sum-up surface array. Thereby, distinguish between different facings. This is
!--       necessary since the routine for a given quantity can be called several times
!--       (for each facing separately). If a surface element does not belong to the currently
!--       treated facing, just add a zero.
          IF ( upward )  THEN
             DO  m = 1, surf_usm%ns
                mean_array(m) = mean_array(m) + MERGE( input_array(m), 0.0_wp, surf_usm%upward(m) )
             ENDDO
          ELSEIF ( downward )  THEN
             DO  m = 1, surf_usm%ns
                mean_array(m) = mean_array(m) + MERGE( input_array(m), 0.0_wp, surf_usm%downward(m) )
             ENDDO
          ELSEIF ( eastward )  THEN
             DO  m = 1, surf_usm%ns
                mean_array(m) = mean_array(m) + MERGE( input_array(m), 0.0_wp, surf_usm%eastward(m) )
             ENDDO
          ELSEIF ( westward )  THEN
             DO  m = 1, surf_usm%ns
                mean_array(m) = mean_array(m) + MERGE( input_array(m), 0.0_wp, surf_usm%westward(m) )
             ENDDO
          ELSEIF ( northward )  THEN
             DO  m = 1, surf_usm%ns
                mean_array(m) = mean_array(m) + MERGE( input_array(m), 0.0_wp, surf_usm%northward(m) )
             ENDDO
          ELSEIF ( southward )  THEN
             DO  m = 1, surf_usm%ns
                mean_array(m) = mean_array(m) + MERGE( input_array(m), 0.0_wp, surf_usm%southward(m) )
             ENDDO
          ENDIF

       ELSEIF ( mode == 'average' )  THEN
!
!--       Average the surface array. Thereby, distinguish between different facings. This is
!--       necessary since the routine for a given quantity can be called several times
!--       (for each facing separately). If a surface element does not belong to the currently
!--       treated facing, just divide it by one.
          IF ( upward )  THEN
             DO  m = 1, surf_usm%ns
                mean_array(m) = mean_array(m) /                                                    &
                          MERGE( REAL( average_count_3d, KIND=wp ), 1.0_wp, surf_usm%upward(m) )
             ENDDO
          ELSEIF ( downward )  THEN
             DO  m = 1, surf_usm%ns
                mean_array(m) = mean_array(m) /                                                    &
                          MERGE( REAL( average_count_3d, KIND=wp ), 1.0_wp, surf_usm%downward(m) )
             ENDDO
          ELSEIF ( eastward )  THEN
             DO  m = 1, surf_usm%ns
                mean_array(m) = mean_array(m) /                                                    &
                          MERGE( REAL( average_count_3d, KIND=wp ), 1.0_wp, surf_usm%eastward(m) )
             ENDDO
          ELSEIF ( westward )  THEN
             DO  m = 1, surf_usm%ns
                mean_array(m) = mean_array(m) /                                                    &
                          MERGE( REAL( average_count_3d, KIND=wp ), 1.0_wp, surf_usm%westward(m) )
             ENDDO
          ELSEIF ( northward )  THEN
             DO  m = 1, surf_usm%ns
                mean_array(m) = mean_array(m) /                                                    &
                          MERGE( REAL( average_count_3d, KIND=wp ), 1.0_wp, surf_usm%northward(m) )
             ENDDO
          ELSEIF ( southward )  THEN
             DO  m = 1, surf_usm%ns
                mean_array(m) = mean_array(m) /                                                    &
                          MERGE( REAL( average_count_3d, KIND=wp ), 1.0_wp, surf_usm%southward(m) )
             ENDDO
          ENDIF
       ENDIF

    END SUBROUTINE average_surfaces

 END SUBROUTINE usm_3d_data_averaging


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Set internal Neumann boundary condition at outer soil grid points for temperature and humidity.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE usm_boundary_condition

    IMPLICIT NONE

    INTEGER(iwp) ::  i      !< grid index x-direction
    INTEGER(iwp) ::  ioff   !< offset index x-direction indicating location of soil grid point
    INTEGER(iwp) ::  j      !< grid index y-direction
    INTEGER(iwp) ::  joff   !< offset index x-direction indicating location of soil grid point
    INTEGER(iwp) ::  k      !< grid index z-direction
    INTEGER(iwp) ::  koff   !< offset index x-direction indicating location of soil grid point
    INTEGER(iwp) ::  m      !< running index surface elements


    DO  m = 1, surf_usm%ns
       ioff = surf_usm%ioff(m)
       joff = surf_usm%joff(m)
       koff = surf_usm%koff(m)
       i = surf_usm%i(m)
       j = surf_usm%j(m)
       k = surf_usm%k(m)
       pt(k+koff,j+joff,i+ioff) = pt(k,j,i)
    ENDDO

 END SUBROUTINE usm_boundary_condition


!--------------------------------------------------------------------------------------------------!
!
! Description:
! ------------
!> Subroutine checks variables and assigns units.
!> It is called out from subroutine check_parameters.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE usm_check_data_output( variable, unit )

    IMPLICIT NONE

    CHARACTER(LEN=*),INTENT(IN)    ::  variable   !<
    CHARACTER(LEN=*),INTENT(OUT)   ::  unit       !<

    CHARACTER(LEN=2)                              ::  ls            !<

    CHARACTER(LEN=varnamelength)                  ::  var           !< TRIM(variable)

    INTEGER(iwp)                                  ::  i,j,l         !< index

    INTEGER(iwp), PARAMETER                       ::  nl1 = 14      !< number of directional usm variables
    CHARACTER(LEN=varnamelength), DIMENSION(nl1)  ::  varlist1 = &  !< list of directional usm variables
              (/'usm_wshf                      ', &
                'usm_wghf                      ', &
                'usm_wghf_window               ', &
                'usm_wghf_green                ', &
                'usm_iwghf                     ', &
                'usm_iwghf_window              ', &
                'usm_surfz                     ', &
                'usm_surfwintrans              ', &
                'usm_surfcat                   ', &
                'usm_t_surf_wall               ', &
                'usm_t_surf_window             ', &
                'usm_t_surf_green              ', &
                'usm_t_green                   ', &
                'usm_qsws                      '/)

    INTEGER(iwp), PARAMETER                       ::  nl2 = 3       !< number of directional layer usm variables
    CHARACTER(LEN=varnamelength), DIMENSION(nl2)  ::  varlist2 = &  !< list of directional layer usm variables
              (/'usm_t_wall                    ', &
                'usm_t_window                  ', &
                'usm_t_green                   '/)

    LOGICAL                                       ::  lfound     !< flag if the variable is found


    lfound = .FALSE.

    var = TRIM( variable )

!
!-- Check if variable exists
!-- Directional variables
    DO  i = 1, nl1
       DO  j = 0, nd-1
          IF ( TRIM( var ) == TRIM( varlist1(i)) // TRIM( dirname(j) ) )  THEN
             lfound = .TRUE.
             EXIT
          ENDIF
          IF ( lfound )  EXIT
       ENDDO
    ENDDO
    IF ( lfound )  GOTO 10
!
!-- Directional layer variables
    DO  i = 1, nl2
       DO  j = 0, nd-1
          DO  l = nzb_wall, nzt_wall
             WRITE( ls,'(A1,I1)' ) '_', l
             IF ( TRIM( var ) == TRIM( varlist2(i) ) // TRIM( ls ) // TRIM( dirname(j) ) )  THEN
                lfound = .TRUE.
                EXIT
             ENDIF
          ENDDO
          IF ( lfound )  EXIT
       ENDDO
    ENDDO
    IF ( .NOT.  lfound )  THEN
       unit = 'illegal'
       RETURN
    ENDIF
10  CONTINUE

    IF ( var(1:9)  == 'usm_wshf_'  .OR.  var(1:9) == 'usm_wghf_' .OR.                              &
         var(1:16) == 'usm_wghf_window_' .OR. var(1:15) == 'usm_wghf_green_' .OR.                  &
         var(1:10) == 'usm_iwghf_' .OR. var(1:17) == 'usm_iwghf_window_'    .OR.                   &
         var(1:17) == 'usm_surfwintrans_' .OR.                                                     &
         var(1:9)  == 'usm_qsws_'  .OR.  var(1:13)  == 'usm_qsws_veg_'  .OR.                       &
         var(1:13) == 'usm_qsws_liq_' )                                                            &
    THEN
       unit = 'W/m2'
    ELSEIF ( var(1:15) == 'usm_t_surf_wall'   .OR.  var(1:10) == 'usm_t_wall' .OR.                 &
             var(1:12) == 'usm_t_window' .OR. var(1:17) == 'usm_t_surf_window' .OR.                &
             var(1:16) == 'usm_t_surf_green'  .OR.                                                 &
             var(1:11) == 'usm_t_green' .OR.  var(1:7) == 'usm_swc' )                              &
    THEN
       unit = 'K'
    ELSEIF ( var(1:9) == 'usm_surfz'  .OR.  var(1:11) == 'usm_surfcat' )  THEN
       unit = '1'
    ELSE
       unit = 'illegal'
    ENDIF

 END SUBROUTINE usm_check_data_output


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Check parameters routine for urban surface model
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE usm_check_parameters

    USE control_parameters,                                                                        &
        ONLY:  bc_pt_b,                                                                            &
               bc_q_b,                                                                             &
               constant_flux_layer,                                                                &
               large_scale_forcing,                                                                &
               lsf_surf,                                                                           &
               topography

    IMPLICIT NONE

    INTEGER(iwp) ::  i  !< running index, x-dimension
    INTEGER(iwp) ::  j  !< running index, y-dimension

!
!-- Dirichlet boundary conditions are required as the surface fluxes are calculated from the
!-- temperature/humidity gradients in the urban surface model
    IF ( bc_pt_b == 'neumann'   .OR.   bc_q_b == 'neumann' )  THEN
       message_string = 'urban surface model requires setting of bc_pt_b = "dirichlet" and '//     &
                        'bc_q_b  = "dirichlet"'
       CALL message( 'usm_check_parameters', 'USM0001', 1, 2, 0, 6, 0 )
    ENDIF

    IF ( .NOT.  constant_flux_layer )  THEN
       message_string = 'urban surface model requires constant_flux_layer = .TRUE.'
       CALL message( 'usm_check_parameters', 'USM0002', 1, 2, 0, 6, 0 )
    ENDIF

    IF (  .NOT.  radiation )  THEN
       message_string = 'urban surface model requires the radiation model to be switched on'
       CALL message( 'usm_check_parameters', 'USM0003', 1, 2, 0, 6, 0 )
    ENDIF
!
!-- Surface forcing has to be disabled for LSF in case of enabled urban surface module
    IF ( large_scale_forcing )  THEN
       lsf_surf = .FALSE.
    ENDIF
!
!-- Topography
    IF ( topography == 'flat' )  THEN
       message_string = 'topography /= "flat" is required when using the urban surface model'
       CALL message( 'usm_check_parameters', 'USM0004', 1, 2, 0, 6, 0 )
    ENDIF
!
!-- Check if building_type is set within a valid range. First, building_type is set via namelist.
    IF ( building_type < LBOUND( building_pars, 2 )  .AND.                                         &
         building_type > UBOUND( building_pars, 2 ) )  THEN
       WRITE( message_string, * ) 'building_type = ', building_type, ' is out of the valid range'
       CALL message( 'usm_check_parameters', 'USM0005', 2, 2, 0, 6, 0 )
    ENDIF
!
!-- building_type is set via the static input file.
    IF ( building_type_f%from_file )  THEN
       DO  i = nxl, nxr
          DO  j = nys, nyn
             IF ( building_type_f%var(j,i) /= building_type_f%fill  .AND.                          &
                  ( building_type_f%var(j,i) < LBOUND( building_pars, 2 )  .OR.                    &
                    building_type_f%var(j,i) > UBOUND( building_pars, 2 ) ) )  THEN
                WRITE( message_string, * ) 'building_type = is out of the valid range at (i,j) ',  &
                                           ' = (', i, ',', j, ')'
                CALL message( 'usm_check_parameters', 'USM0006', 2, 2, myid, 6, 0 )
             ENDIF
          ENDDO
       ENDDO
    ENDIF
!
!-- Check if building_pars is correctly dimensioned.
    IF ( building_pars_f%from_file )  THEN
       IF ( SIZE( building_pars_f%pars ) /= SIZE( building_pars, 1 ) )  THEN
          WRITE( message_string, * ) 'dimension size of static input variable building_pars is ',  &
                                     SIZE( building_pars_f%pars ), '&',                            &
                                     'dimension size of ', SIZE( building_pars, 1 ), 'is required'
          CALL message( 'usm_check_parameters', 'USM0007', 2, 2, 0, 6, 0 )
       ENDIF
    ENDIF

 END SUBROUTINE usm_check_parameters


!--------------------------------------------------------------------------------------------------!
!
! Description:
! ------------
!> Output of the 3D-arrays in netCDF and/or AVS format for variables of urban_surface model.
!> It resorts the urban surface module output quantities from surf style indexing into temporary 3D
!> array with indices (i,j,k). It is called from subroutine data_output_3d.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE usm_data_output_3d( av, variable, found, local_pf, nzb_do, nzt_do )

    IMPLICIT NONE

    CHARACTER(LEN=*), INTENT(IN)   ::  variable  !< variable name

    CHARACTER(LEN=varnamelength)   ::  var  !< trimmed variable name

    INTEGER(iwp), INTENT(IN)       ::  av        !< flag if averaged
    INTEGER(iwp), INTENT(IN)       ::  nzb_do    !< lower limit of the data output (usually 0)
    INTEGER(iwp), INTENT(IN)       ::  nzt_do    !< vertical upper limit of the data output (usually nz_do3d)

    INTEGER(iwp)  ::  ids, idsint, idsidx        !<
    INTEGER(iwp)  ::  i, j, k, iwl, istat, m     !< running indices

    LOGICAL ::  downward  !< control flag indicating the output of downward-facing surfaces
    LOGICAL ::  eastward  !< control flag indicating the output of east-facing surfaces
    LOGICAL ::  northward !< control flag indicating the output of northward-facing surfaces
    LOGICAL ::  southward !< control flag indicating the output of southward-facing surfaces
    LOGICAL ::  upward    !< control flag indicating the output of upward-facing surfaces
    LOGICAL ::  westward  !< control flag indicating the output of westward-facing surfaces

    LOGICAL, INTENT(OUT) ::  found  !<

    REAL(wp), DIMENSION(nxl:nxr,nys:nyn,nzb_do:nzt_do) ::  local_pf  !< sp - it has to correspond to module data_output_3d
    REAL(wp), DIMENSION(nzb:nzt+1,nys:nyn,nxl:nxr)     ::  temp_pf   !< temp array for urban surface output procedure


    found = .TRUE.
    temp_pf = output_fill_value

    ids = -1
    var = TRIM( variable )
    DO i = 0, nd-1
        k = len( TRIM( var ) )
        j = len( TRIM( dirname(i) ) )
        IF ( TRIM( var(k-j+1:k) ) == TRIM( dirname(i) ) )  THEN
            ids = i
            idsint = dirint(ids)
            idsidx = diridx(ids)
            var = var(:k-j)
            EXIT
        ENDIF
    ENDDO
!
!-- Set direction control flags
    downward  = .FALSE.
    eastward  = .FALSE.
    northward = .FALSE.
    southward = .FALSE.
    upward    = .FALSE.
    westward  = .FALSE.
    IF ( idsint == iup )  THEN
       upward = .TRUE.
    ELSEIF ( idsint == idown )  THEN
       downward = .TRUE.
    ELSEIF ( idsint == ieast )  THEN
       eastward = .TRUE.
    ELSEIF ( idsint == iwest )  THEN
       westward = .TRUE.
    ELSEIF ( idsint == inorth )  THEN
       northward = .TRUE.
    ELSEIF ( idsint == isouth )  THEN
       southward = .TRUE.
    ENDIF

    IF ( ids == -1 )  THEN
        var = TRIM( variable )
    ENDIF
    IF ( var(1:11) == 'usm_t_wall_'  .AND.  len( TRIM( var ) ) >= 12 )  THEN
!
!--     Wall layers
        READ( var(12:12), '(I1)', iostat = istat ) iwl
        IF ( istat == 0  .AND.  iwl >= nzb_wall  .AND.  iwl <= nzt_wall )  THEN
            var = var(1:10)
        ENDIF
    ENDIF
    IF ( var(1:13) == 'usm_t_window_'  .AND.  len( TRIM( var ) ) >= 14 )  THEN
!
!--     Window layers
        READ( var(14:14), '(I1)', iostat = istat ) iwl
        IF ( istat == 0  .AND.  iwl >= nzb_wall  .AND.  iwl <= nzt_wall )  THEN
            var = var(1:12)
        ENDIF
    ENDIF
    IF ( var(1:12) == 'usm_t_green_'  .AND.  len( TRIM( var ) ) >= 13 )  THEN
!
!--     Green layers
        READ( var(13:13), '(I1)', iostat = istat ) iwl
        IF ( istat == 0  .AND.  iwl >= nzb_wall  .AND.  iwl <= nzt_wall )  THEN
            var = var(1:11)
        ENDIF
    ENDIF
    IF ( var(1:8) == 'usm_swc_'  .AND.  len( TRIM( var ) ) >= 9 )  THEN
!
!--     Green layers soil water content
        READ( var(9:9), '(I1)', iostat = istat ) iwl
        IF ( istat == 0  .AND.  iwl >= nzb_wall  .AND.  iwl <= nzt_wall )  THEN
            var = var(1:7)
        ENDIF
    ENDIF

    SELECT CASE ( TRIM( var ) )

       CASE ( 'usm_surfz' )
!
!--       Array of surface height (z)
!--       Write surface array to temp_pf. Thereby, distinguish between different facings. This is
!--       necessary since the routine for a given quantity can be called several times
!--       (for each facing separately). If a surface element does not belong to the currently
!--       treated facing, do not modify temp_pf's current value.
          IF ( upward )  THEN
             DO  m = 1, surf_usm%ns
                i = surf_usm%i(m)
                j = surf_usm%j(m)
                k = surf_usm%k(m)
                temp_pf(k,j,i) = MERGE( MAX( temp_pf(0,j,i), REAL( k, KIND = wp) ),                &
                                        temp_pf(k,j,i), surf_usm%upward(m) )
             ENDDO
          ELSEIF ( downward )  THEN
             DO  m = 1, surf_usm%ns
                i = surf_usm%i(m)
                j = surf_usm%j(m)
                k = surf_usm%k(m)
                temp_pf(k,j,i) = MERGE( MAX( temp_pf(0,j,i), REAL( k, KIND = wp) ),                &
                                        temp_pf(k,j,i), surf_usm%downward(m) )
             ENDDO
          ELSEIF ( eastward )  THEN
             DO  m = 1, surf_usm%ns
                i = surf_usm%i(m)
                j = surf_usm%j(m)
                k = surf_usm%k(m)
                temp_pf(k,j,i) = MERGE( MAX( temp_pf(0,j,i), REAL( k, KIND = wp) + 1.0_sp ),      &
                                        temp_pf(k,j,i), surf_usm%eastward(m) )
             ENDDO
          ELSEIF ( westward )  THEN
             DO  m = 1, surf_usm%ns
                i = surf_usm%i(m)
                j = surf_usm%j(m)
                k = surf_usm%k(m)
                temp_pf(k,j,i) = MERGE( MAX( temp_pf(0,j,i), REAL( k, KIND = wp) + 1.0_sp ),      &
                                        temp_pf(k,j,i), surf_usm%westward(m) )
             ENDDO
          ELSEIF ( northward )  THEN
             DO  m = 1, surf_usm%ns
                i = surf_usm%i(m)
                j = surf_usm%j(m)
                k = surf_usm%k(m)
                temp_pf(k,j,i) = MERGE( MAX( temp_pf(0,j,i), REAL( k, KIND = wp) + 1.0_sp ),      &
                                        temp_pf(k,j,i), surf_usm%northward(m) )
             ENDDO
          ELSEIF ( southward )  THEN
             DO  m = 1, surf_usm%ns
                i = surf_usm%i(m)
                j = surf_usm%j(m)
                k = surf_usm%k(m)
                temp_pf(k,j,i) = MERGE( MAX( temp_pf(0,j,i), REAL( k, KIND = wp) + 1.0_sp ),      &
                                        temp_pf(k,j,i), surf_usm%southward(m) )
             ENDDO
          ENDIF

       CASE ( 'usm_surfcat' )
!
!--       Surface category
          CALL write_surface_data_to_temp_pf( REAL( surf_usm%surface_types, KIND = wp ) )

       CASE ( 'usm_surfwintrans' )
!
!--       Transmissivity window tiles
          CALL write_surface_data_to_temp_pf( surf_usm%transmissivity )


       CASE ( 'usm_wshf' )
!
!--       Sensible heat flux
          IF ( av == 0 )  THEN
             CALL write_surface_data_to_temp_pf( surf_usm%wshf_eb )
          ELSE
             CALL write_surface_data_to_temp_pf( wshf_eb_av%val )
          ENDIF

       CASE ( 'usm_qsws' )
!
!--       Latent heat flux
          IF ( av == 0 )  THEN
             CALL write_surface_data_to_temp_pf( surf_usm%qsws )
          ELSE
             CALL write_surface_data_to_temp_pf( qsws_av%val )
          ENDIF

       CASE ( 'usm_qsws_veg' )
!
!--       Latent heat flux from vegetation surfaces
          IF ( av == 0 )  THEN
             CALL write_surface_data_to_temp_pf( surf_usm%qsws_veg )
          ELSE
             CALL write_surface_data_to_temp_pf( qsws_veg_av%val )
          ENDIF

       CASE ( 'usm_qsws_liq' )
!
!--       Latent heat flux from surfaces with liquid
          IF ( av == 0 )  THEN
             CALL write_surface_data_to_temp_pf( surf_usm%qsws_liq )
          ELSE
             CALL write_surface_data_to_temp_pf( qsws_liq_av%val )
          ENDIF

       CASE ( 'usm_wghf' )
!
!--       Heat flux from ground
          IF ( av == 0 )  THEN
             CALL write_surface_data_to_temp_pf( surf_usm%wghf_eb )
          ELSE
             CALL write_surface_data_to_temp_pf( wghf_eb_av%val )
          ENDIF

       CASE ( 'usm_wghf_window' )
!
!--       Heat flux from window ground
          IF ( av == 0 )  THEN
             CALL write_surface_data_to_temp_pf( surf_usm%wghf_eb_window )
          ELSE
             CALL write_surface_data_to_temp_pf( wghf_eb_window_av%val )
          ENDIF

       CASE ( 'usm_wghf_green' )
!
!--       Heat flux from green ground
          IF ( av == 0 )  THEN
             CALL write_surface_data_to_temp_pf( surf_usm%wghf_eb_green )
          ELSE
             CALL write_surface_data_to_temp_pf( wghf_eb_green_av%val )
          ENDIF

       CASE ( 'usm_iwghf' )
!
!--       Heat flux from indoor ground
          IF ( av == 0 )  THEN
             CALL write_surface_data_to_temp_pf( surf_usm%iwghf_eb )
          ELSE
             CALL write_surface_data_to_temp_pf( iwghf_eb_av%val )
          ENDIF

       CASE ( 'usm_iwghf_window' )
!
!--       Heat flux from indoor window ground
          IF ( av == 0 )  THEN
             CALL write_surface_data_to_temp_pf( surf_usm%iwghf_eb_window )
          ELSE
             CALL write_surface_data_to_temp_pf( iwghf_eb_window_av%val )
          ENDIF

       CASE ( 'usm_t_surf_wall' )
!
!--       Surface temperature for wall surfaces
          IF ( av == 0 )  THEN
             CALL write_surface_data_to_temp_pf( t_surf_wall%val )
          ELSE
             CALL write_surface_data_to_temp_pf( t_surf_wall_av%val )
          ENDIF

       CASE ( 'usm_t_surf_window' )
!
!--       Surface temperature for window surfaces
          IF ( av == 0 )  THEN
             CALL write_surface_data_to_temp_pf( t_surf_window%val )
          ELSE
             CALL write_surface_data_to_temp_pf( t_surf_window_av%val )
          ENDIF

       CASE ( 'usm_t_surf_green' )
!
!--       Surface temperature for green surfaces
          IF ( av == 0 )  THEN
             CALL write_surface_data_to_temp_pf( t_surf_green%val )
          ELSE
             CALL write_surface_data_to_temp_pf( t_surf_green_av%val )
          ENDIF

       CASE ( 'usm_t_wall' )
!
!--       Wall temperature for iwl layer of walls
          IF ( av == 0 )  THEN
             CALL write_surface_data_to_temp_pf( t_wall%val(iwl,:) )
          ELSE
             CALL write_surface_data_to_temp_pf( t_wall_av%val(iwl,:) )
          ENDIF

       CASE ( 'usm_t_window' )
!
!--       Window temperature for iwl layer
          IF ( av == 0 )  THEN
             CALL write_surface_data_to_temp_pf( t_window%val(iwl,:) )
          ELSE
             CALL write_surface_data_to_temp_pf( t_window_av%val(iwl,:) )
          ENDIF

       CASE ( 'usm_t_green' )
!
!--       Green temperature for iwl layer
          IF ( av == 0 )  THEN
             CALL write_surface_data_to_temp_pf( t_green%val(iwl,:) )
          ELSE
             CALL write_surface_data_to_temp_pf( t_green_av%val(iwl,:) )
          ENDIF

       CASE ( 'usm_swc' )
!
!--       Soil water content for iwl layer
          IF ( av == 0 )  THEN
             CALL write_surface_data_to_temp_pf( swc%val(iwl,:) )
          ELSE
             CALL write_surface_data_to_temp_pf( swc_av%val(iwl,:) )
          ENDIF

       CASE DEFAULT
          found = .FALSE.
          RETURN

    END SELECT

!
!-- Rearrange dimensions for NetCDF output
!-- FIXME: this may generate FPE overflow upon conversion from DP to SP
    DO  j = nys, nyn
        DO  i = nxl, nxr
            DO  k = nzb_do, nzt_do
                local_pf(i,j,k) = temp_pf(k,j,i)
            ENDDO
        ENDDO
    ENDDO

    CONTAINS

!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Write surface data onto output array accorrding to its facing.
!--------------------------------------------------------------------------------------------------!
    SUBROUTINE write_surface_data_to_temp_pf( surf_array )

       REAL(wp), DIMENSION(1:surf_usm%ns) ::  surf_array !< treated surface array
!
!--    Write surface array to temp_pf. Thereby, distinguish between different facings. This is
!--    necessary since the routine for a given quantity can be called several times
!--    (for each facing separately). If a surface element does not belong to the currently
!--    treated facing, do not modify temp_pf's current value.
       IF ( upward )  THEN
          DO  m = 1, surf_usm%ns
             i = surf_usm%i(m)
             j = surf_usm%j(m)
             k = surf_usm%k(m)
             temp_pf(k,j,i) = MERGE( surf_array(m), temp_pf(k,j,i), surf_usm%upward(m) )
          ENDDO
       ELSEIF ( downward )  THEN
          DO  m = 1, surf_usm%ns
             i = surf_usm%i(m)
             j = surf_usm%j(m)
             k = surf_usm%k(m)
             temp_pf(k,j,i) = MERGE( surf_array(m), temp_pf(k,j,i), surf_usm%downward(m) )
          ENDDO
       ELSEIF ( eastward )  THEN
          DO  m = 1, surf_usm%ns
             i = surf_usm%i(m)
             j = surf_usm%j(m)
             k = surf_usm%k(m)
             temp_pf(k,j,i) = MERGE( surf_array(m), temp_pf(k,j,i), surf_usm%eastward(m) )
          ENDDO
       ELSEIF ( westward )  THEN
          DO  m = 1, surf_usm%ns
             i = surf_usm%i(m)
             j = surf_usm%j(m)
             k = surf_usm%k(m)
             temp_pf(k,j,i) = MERGE( surf_array(m), temp_pf(k,j,i), surf_usm%westward(m) )
          ENDDO
       ELSEIF ( northward )  THEN
          DO  m = 1, surf_usm%ns
             i = surf_usm%i(m)
             j = surf_usm%j(m)
             k = surf_usm%k(m)
             temp_pf(k,j,i) = MERGE( surf_array(m), temp_pf(k,j,i), surf_usm%northward(m) )
          ENDDO
       ELSEIF ( southward )  THEN
          DO  m = 1, surf_usm%ns
             i = surf_usm%i(m)
             j = surf_usm%j(m)
             k = surf_usm%k(m)
             temp_pf(k,j,i) = MERGE( surf_array(m), temp_pf(k,j,i), surf_usm%southward(m) )
          ENDDO
       ENDIF

    END SUBROUTINE write_surface_data_to_temp_pf

 END SUBROUTINE usm_data_output_3d


!--------------------------------------------------------------------------------------------------!
!
! Description:
! ------------
!> Soubroutine defines appropriate grid for netcdf variables.
!> It is called out from subroutine netcdf.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE usm_define_netcdf_grid( variable, found, grid_x, grid_y, grid_z )

    IMPLICIT NONE

    CHARACTER(LEN=*), INTENT(IN)  ::  variable  !<
    CHARACTER(LEN=*), INTENT(OUT) ::  grid_x    !<
    CHARACTER(LEN=*), INTENT(OUT) ::  grid_y    !<
    CHARACTER(LEN=*), INTENT(OUT) ::  grid_z    !<

    CHARACTER(LEN=varnamelength)  ::  var  !<

    LOGICAL, INTENT(OUT)  ::  found  !<

    var = TRIM( variable )
    IF ( var(1:9) == 'usm_wshf_'  .OR.  var(1:9) == 'usm_wghf_'  .OR.                              &
         var(1:16) == 'usm_wghf_window_'  .OR. var(1:15) == 'usm_wghf_green_' .OR.                 &
         var(1:10) == 'usm_iwghf_'  .OR. var(1:17) == 'usm_iwghf_window_' .OR.                     &
         var(1:9) == 'usm_qsws_'  .OR.  var(1:13) == 'usm_qsws_veg_'  .OR.                         &
         var(1:13) == 'usm_qsws_liq_' .OR.                                                         &
         var(1:15) == 'usm_t_surf_wall'  .OR.  var(1:10) == 'usm_t_wall'  .OR.                     &
         var(1:17) == 'usm_t_surf_window'  .OR.  var(1:12) == 'usm_t_window'  .OR.                 &
         var(1:16) == 'usm_t_surf_green'  .OR. var(1:11) == 'usm_t_green' .OR.                     &
         var(1:9) == 'usm_surfz'  .OR.  var(1:11) == 'usm_surfcat'  .OR.                           &
         var(1:16) == 'usm_surfwintrans'  .OR. var(1:7) == 'usm_swc' ) THEN

        found = .TRUE.
        grid_x = 'x'
        grid_y = 'y'
        grid_z = 'zu'
    ELSE
        found  = .FALSE.
        grid_x = 'none'
        grid_y = 'none'
        grid_z = 'none'
    ENDIF

 END SUBROUTINE usm_define_netcdf_grid


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Initialization of the wall surface model
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE usm_init_wall_heat_model

    IMPLICIT NONE

    INTEGER(iwp) ::  k  !< running index along z-dimension
    INTEGER(iwp) ::  m  !< running index for surface elements


    IF ( debug_output )  CALL debug_message( 'usm_init_wall_heat_model', 'start' )

!
!-- Calculate wall and window grid spacings. Wall temperature is defined at the center of the
!-- wall layers.
    DO  m = 1, surf_usm%ns
!
!--    Set-up wall layer discretization
       surf_usm%dz_wall(nzb_wall,m) = surf_usm%zw(nzb_wall,m)
       DO k = nzb_wall+1, nzt_wall
          surf_usm%dz_wall(k,m) = surf_usm%zw(k,m) - surf_usm%zw(k-1,m)
       ENDDO

       DO  k = nzb_wall, nzt_wall-1
          surf_usm%dz_wall_center(k,m) = 0.5_wp *                                                  &
                                         ( surf_usm%dz_wall(k,m) + surf_usm%dz_wall(k+1,m) )
          IF ( surf_usm%dz_wall_center(k,m) <= 0.0_wp )  THEN
             WRITE ( message_string, '(A,I5,A)' ) 'invalid wall layer configuration found: ' //    &
                                                  'dz_wall_center(k=', k, ') <= 0.0'
             CALL message( 'usm_init_wall_heat_model', 'USM0008', 1, 2, 0, 6, 0 )
          ENDIF
       ENDDO
       surf_usm%dz_wall_center(nzt_wall,m) = surf_usm%dz_wall(nzt_wall,m)

!
!--    Set-up window layer discretization
       surf_usm%dz_window(nzb_wall,m) = surf_usm%zw_window(nzb_wall,m)
       DO  k = nzb_wall+1, nzt_wall
          surf_usm%dz_window(k,m) = surf_usm%zw_window(k,m) - surf_usm%zw_window(k-1,m)
       ENDDO

       DO  k = nzb_wall, nzt_wall-1
          surf_usm%dz_window_center(k,m) = 0.5_wp *                                                &
                                           ( surf_usm%dz_window(k,m) + surf_usm%dz_window(k+1,m) )
          IF ( surf_usm%dz_window_center(k,m) <= 0.0_wp )  THEN
             WRITE ( message_string, '(A,I5,A)' ) 'invalid window layer configuration found: ' //  &
                                                  'dz_window_center(k=', k, ') <= 0.0'
             CALL message( 'usm_init_wall_heat_model', 'USM0009', 1, 2, 0, 6, 0 )
          ENDIF
       ENDDO

       surf_usm%dz_window_center(nzt_wall,m) = surf_usm%dz_window(nzt_wall,m)

!
!--    Set-up green roofs
       IF (surf_usm%green_type_roof(m) == 2.0_wp )  THEN
!
!--       Extensive green roof
!--       Set ratio of substrate layer thickness, soil-type and LAI
          soil_type = 3
          surf_usm%lai(m) = 2.0_wp
          surf_usm%zw_green(nzb_wall,m)   = 0.05_wp
          surf_usm%zw_green(nzb_wall+1,m) = 0.10_wp
          surf_usm%zw_green(nzb_wall+2,m) = 0.15_wp
          surf_usm%zw_green(nzb_wall+3,m) = 0.20_wp
       ELSE
!
!--       Intensive green roof
!--       Set ratio of substrate layer thickness, soil-type and LAI
          soil_type = 6
          surf_usm%lai(m) = 4.0_wp
          surf_usm%zw_green(nzb_wall,m)   = 0.05_wp
          surf_usm%zw_green(nzb_wall+1,m) = 0.10_wp
          surf_usm%zw_green(nzb_wall+2,m) = 0.40_wp
          surf_usm%zw_green(nzb_wall+3,m) = 0.80_wp
       ENDIF

       surf_usm%dz_green(nzb_wall,m) = surf_usm%zw_green(nzb_wall,m)
       DO k = nzb_wall+1, nzt_wall
           surf_usm%dz_green(k,m) = surf_usm%zw_green(k,m) - surf_usm%zw_green(k-1,m)
       ENDDO


       DO  k = nzb_wall, nzt_wall-1
          surf_usm%dz_green_center(k,m) = 0.5_wp *                                                 &
                                          ( surf_usm%dz_green(k,m) + surf_usm%dz_green(k+1,m) )
          IF ( surf_usm%dz_green_center(k,m) <= 0.0_wp )  THEN
             WRITE ( message_string, '(A,I5,A)' ) 'invalid green layer configuration found: ' //   &
                                                  'dz_green_center(k=', k, ') <= 0.0'
             CALL message( 'usm_init_wall_heat_model', 'USM0010', 1, 2, 0, 6, 0 )
          ENDIF
       ENDDO

       surf_usm%dz_green_center(nzt_wall,m) = surf_usm%dz_green(nzt_wall,m)

       IF ( alpha_vangenuchten == 9999999.9_wp )  THEN
          alpha_vangenuchten = soil_pars(0,soil_type)
       ENDIF

       IF ( l_vangenuchten == 9999999.9_wp )  THEN
          l_vangenuchten = soil_pars(1,soil_type)
       ENDIF

       IF ( n_vangenuchten == 9999999.9_wp )  THEN
          n_vangenuchten = soil_pars(2,soil_type)
       ENDIF

       IF ( hydraulic_conductivity == 9999999.9_wp )  THEN
          hydraulic_conductivity = soil_pars(3,soil_type)
       ENDIF

       IF ( saturation_moisture == 9999999.9_wp )  THEN
          saturation_moisture = m_soil_pars(0,soil_type)
       ENDIF

       IF ( field_capacity == 9999999.9_wp )  THEN
          field_capacity = m_soil_pars(1,soil_type)
       ENDIF

       IF ( wilting_point == 9999999.9_wp )  THEN
          wilting_point = m_soil_pars(2,soil_type)
       ENDIF

       IF ( residual_moisture == 9999999.9_wp )  THEN
          residual_moisture = m_soil_pars(3,soil_type)
       ENDIF

       IF ( TRIM( initializing_actions ) /= 'read_restart_data'  .AND.  .NOT. read_spinup_data )   &
       THEN
          DO  k = nzb_wall, nzt_wall+1
             swc%val(k,m) = field_capacity
          ENDDO
       ENDIF

       DO  k = nzb_wall, nzt_wall+1
          rootfr%val(k,m)                 = 0.5_wp
          surf_usm%alpha_vg_green(m)      = alpha_vangenuchten
          surf_usm%l_vg_green(m)          = l_vangenuchten
          surf_usm%n_vg_green(m)          = n_vangenuchten
          surf_usm%gamma_w_green_sat(k,m) = hydraulic_conductivity
          swc_sat%val(k,m)                = saturation_moisture
          fc%val(k,m)                     = field_capacity
          wilt%val(k,m)                   = wilting_point
          swc_res%val(k,m)                = residual_moisture
       ENDDO

    ENDDO

    surf_usm%ddz_wall          = 1.0_wp / surf_usm%dz_wall
    surf_usm%ddz_wall_center   = 1.0_wp / surf_usm%dz_wall_center
    surf_usm%ddz_window        = 1.0_wp / surf_usm%dz_window
    surf_usm%ddz_window_center = 1.0_wp / surf_usm%dz_window_center
    surf_usm%ddz_green         = 1.0_wp / surf_usm%dz_green
    surf_usm%ddz_green_center  = 1.0_wp / surf_usm%dz_green_center

!
!-- Calculate wall heat conductivity (lambda_h) at the _layer level the weighted average
    DO  m = 1, surf_usm%ns
       DO  k = nzb_wall, nzt_wall-1
          surf_usm%lambda_h_layer(k,m) = ( surf_usm%lambda_h(k,m)   * surf_usm%dz_wall(k,m) +      &
                                           surf_usm%lambda_h(k+1,m) * surf_usm%dz_wall(k+1,m)      &
                                         ) * 0.5_wp * surf_usm%ddz_wall_center(k,m)
       ENDDO
       surf_usm%lambda_h_layer(nzt_wall,m) = surf_usm%lambda_h(nzt_wall,m)
    ENDDO

    DO  m = 1, surf_usm%ns
!
!--    Calculate wall heat conductivity (lambda_h) at the _layer level using weighting
       DO  k = nzb_wall, nzt_wall-1
          surf_usm%lambda_h_window_layer(k,m) = ( surf_usm%lambda_h_window(k,m)   *                &
                                                  surf_usm%dz_window(k,m) +                        &
                                                  surf_usm%lambda_h_window(k+1,m) *                &
                                                  surf_usm%dz_window(k+1,m)                        &
                                                ) * 0.5_wp * surf_usm%ddz_window_center(k,m)
       ENDDO
       surf_usm%lambda_h_window_layer(nzt_wall,m) = surf_usm%lambda_h_window(nzt_wall,m)
    ENDDO

    IF ( debug_output )  CALL debug_message( 'usm_init_wall_heat_model', 'end' )

 END SUBROUTINE usm_init_wall_heat_model


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Initialization of the urban surface model
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE usm_init

    USE arrays_3d,                                                                                 &
        ONLY:  zw

    IMPLICIT NONE

    INTEGER(iwp) ::  exit_index        !< to store surface element index where z0 limit is exceeded
    INTEGER(iwp) ::  i                 !< loop index x-dirction
    INTEGER(iwp) ::  ind_alb_green     !< index in input list for green albedo
    INTEGER(iwp) ::  ind_alb_wall      !< index in input list for wall albedo
    INTEGER(iwp) ::  ind_alb_win       !< index in input list for window albedo
    INTEGER(iwp) ::  ind_emis_wall     !< index in input list for wall emissivity
    INTEGER(iwp) ::  ind_emis_green    !< index in input list for green emissivity
    INTEGER(iwp) ::  ind_emis_win      !< index in input list for window emissivity
    INTEGER(iwp) ::  ind_green_frac_w  !< index in input list for green fraction on wall
    INTEGER(iwp) ::  ind_green_frac    !< index in input list for green fraction on roof
    INTEGER(iwp) ::  ind_hc1           !< index in input list for heat capacity at first wall layer
    INTEGER(iwp) ::  ind_hc1_win       !< index in input list for heat capacity at first window layer
    INTEGER(iwp) ::  ind_hc2           !< index in input list for heat capacity at second wall layer
    INTEGER(iwp) ::  ind_hc2_win       !< index in input list for heat capacity at second window layer
    INTEGER(iwp) ::  ind_hc3           !< index in input list for heat capacity at third wall layer
    INTEGER(iwp) ::  ind_hc3_win       !< index in input list for heat capacity at third window layer
    INTEGER(iwp) ::  ind_hc4           !< index in input list for heat capacity at fourth wall layer
    INTEGER(iwp) ::  ind_hc4_win       !< index in input list for heat capacity at fourth window layer
    INTEGER(iwp) ::  ind_lai           !< index in input list for LAI on roof
    INTEGER(iwp) ::  ind_lai_w         !< index in input list for LAI on wall
    INTEGER(iwp) ::  ind_tc1           !< index in input list for thermal conductivity at first wall layer
    INTEGER(iwp) ::  ind_tc1_win       !< index in input list for thermal conductivity at first window layer
    INTEGER(iwp) ::  ind_tc2           !< index in input list for thermal conductivity at second wall layer
    INTEGER(iwp) ::  ind_tc2_win       !< index in input list for thermal conductivity at second window layer
    INTEGER(iwp) ::  ind_tc3           !< index in input list for thermal conductivity at third wall layer
    INTEGER(iwp) ::  ind_tc3_win       !< index in input list for thermal conductivity at third window layer
    INTEGER(iwp) ::  ind_tc4           !< index in input list for thermal conductivity at fourth wall layer
    INTEGER(iwp) ::  ind_tc4_win       !< index in input list for thermal conductivity at fourth window layer
    INTEGER(iwp) ::  ind_thick_1       !< index in input list for thickness of first wall layer
    INTEGER(iwp) ::  ind_thick_1_win   !< index in input list for thickness of first window layer
    INTEGER(iwp) ::  ind_thick_2       !< index in input list for thickness of second wall layer
    INTEGER(iwp) ::  ind_thick_2_win   !< index in input list for thickness of second window layer
    INTEGER(iwp) ::  ind_thick_3       !< index in input list for thickness of third wall layer
    INTEGER(iwp) ::  ind_thick_3_win   !< index in input list for thickness of third window layer
    INTEGER(iwp) ::  ind_thick_4       !< index in input list for thickness of fourth wall layer
    INTEGER(iwp) ::  ind_thick_4_win   !< index in input list for thickness of fourth window layer
    INTEGER(iwp) ::  ind_trans         !< index in input list for window transmissivity
    INTEGER(iwp) ::  ind_wall_frac     !< index in input list for wall fraction
    INTEGER(iwp) ::  ind_win_frac      !< index in input list for window fraction
    INTEGER(iwp) ::  ind_z0            !< index in input list for z0
    INTEGER(iwp) ::  ind_z0qh          !< index in input list for z0h / z0q
    INTEGER(iwp) ::  is                !< loop index input surface element
    INTEGER(iwp) ::  j                 !< loop index y-dirction
    INTEGER(iwp) ::  k                 !< loop index z-dirction
    INTEGER(iwp) ::  m                 !< loop index surface element
    INTEGER(iwp) ::  st                !< dummy

    LOGICAL ::  flag_exceed_z0           !< dummy flag to indicate whether roughness length is too high
    LOGICAL ::  flag_exceed_z0h          !< dummy flag to indicate whether roughness length for temperature is too high
    LOGICAL ::  flag_exceed_z0q          !< dummy flag to indicate whether roughness length for mositure is too high
    LOGICAL ::  relative_fraction_error  !< flag indicating if relative surface fractions do not sum up to 1

    REAL(wp)     ::  c                     !<
    REAL(wp)     ::  ground_floor_level_l  !< local height of ground floor level
    REAL(wp)     ::  tin                   !<
    REAL(wp)     ::  twin                  !<
    REAL(wp)     ::  z_agl                 !< height of the surface element above terrain


    IF ( debug_output )  CALL debug_message( 'usm_init', 'start' )

    CALL cpu_log( log_point_s(78), 'usm_init', 'start' )
!
!-- Surface forcing has to be disabled for LSF in case of enabled urban surface module
    IF ( large_scale_forcing )  THEN
        lsf_surf = .FALSE.
    ENDIF
!
!-- Store wall indices on surface structure.
    surf_usm%nzb_wall = nzb_wall
    surf_usm%nzt_wall = nzt_wall
!
!-- Calculate constant values
    d_roughness_concrete = 1.0_wp / roughness_concrete
!
!-- Flag surface elements belonging to the ground floor level. Therefore, use terrain height array
!-- from file, if available. This flag is later used to control initialization of surface attributes.
    surf_usm%gfl = .FALSE.

    DO  m = 1, surf_usm%ns
       i = surf_usm%i(m) + surf_usm%ioff(m)
       j = surf_usm%j(m) + surf_usm%joff(m)
       k = surf_usm%k(m)
!
!--    Determine local ground level. Level 1 - default value, level 2 - initialization according
!--    to building type, level 3 - initialization from value read from file.
       ground_floor_level_l = ground_floor_level

       IF ( building_type_f%from_file )  THEN
           ground_floor_level_l = building_pars(ind_gflh,building_type_f%var(j,i))
       ENDIF

       IF ( building_pars_f%from_file )  THEN
          IF ( building_pars_f%pars_xy(ind_gflh,j,i) /=  building_pars_f%fill )                    &
             ground_floor_level_l = building_pars_f%pars_xy(ind_gflh,j,i)
       ENDIF
!
!--    Determine height of surface element above ground level. Please note, the height of a
!--    surface element is determined with respect to its height above ground of the reference
!--    grid point in the atmosphere. As (j,i) are defined as the surface indices and not
!--    the reference grid point, substract the offset values when assessing the terrain height.
       IF ( terrain_height_f%from_file )  THEN
          z_agl = zw(k) - terrain_height_f%var(j-surf_usm%joff(m),i-surf_usm%ioff(m))
       ELSE
          z_agl = zw(k)
       ENDIF
!
!--    Set flag for ground level
       IF ( z_agl <= ground_floor_level_l ) surf_usm%gfl(m) = .TRUE.
!
!--    Set ground-floor level attribute to False at horizontally upward-facing surfaces.
!--    This is because ground-floor level properties are not necessarily valid for roofs,
!--    only valid for facades.
       IF ( surf_usm%upward(m) )  surf_usm%gfl(m) = .FALSE.
    ENDDO
!
!-- Initialization of resistances.
    DO  m = 1, surf_usm%ns
       surf_usm%r_a(m)        = 50.0_wp
       surf_usm%r_a_green(m)  = 50.0_wp
       surf_usm%r_a_window(m) = 50.0_wp
    ENDDO
!
!-- Initialization of canopy properties
    DO  m = 1, surf_usm%ns
       surf_usm%r_canopy(m)     = 200.0_wp !< canopy_resistance
       surf_usm%r_canopy_min(m) = 200.0_wp !< min_canopy_resistance
       surf_usm%g_d(m)          = 0.0_wp   !< canopy_resistance_coefficient
    ENDDO
!
!--  Initialize urban-type surface attributes. According to initialization in land-surface model,
!--  follow a 3-level approach.
!--  Level 1 - initialization of roofs (upward-facing) via default attributes. Note, for sake
!--  of clearness, initialization of roofs (normal vector component in z) and walls is
!--  separated.
     DO  m = 1, surf_usm%ns
        IF ( surf_usm%upward(m)  .OR.  surf_usm%downward(m) )  THEN
!
!--        Now, all horizontal surfaces are roof surfaces (?)
           IF ( surf_usm%upward(m) )  surf_usm%isroof_surf(m)   = .TRUE.
           IF ( surf_usm%upward(m) )  surf_usm%surface_types(m) = roof_category !< default category for root surface
!
!--        In order to distinguish between ground floor level and above-ground-floor level surfaces,
!--        set input indices.

           ind_green_frac = ind_green_frac_r_agfl
           ind_lai        = ind_lai_r_agfl
           ind_z0         = ind_z0_agfl
           ind_z0qh       = ind_z0qh_agfl
!
!--        Store building type and its name on each surface element
           surf_usm%building_type(m)      = building_type
           surf_usm%building_type_name(m) = building_type_name(building_type)
!
!--        Initialize relatvie wall- (0), green- (1) and window (2) fractions
           surf_usm%frac(m,ind_veg_wall)  = building_pars(ind_wall_frac_r,building_type)
           surf_usm%frac(m,ind_pav_green) = building_pars(ind_green_frac,building_type)
           surf_usm%frac(m,ind_wat_win)   = building_pars(ind_win_frac_r,building_type)
           surf_usm%lai(m)                = building_pars(ind_lai,building_type)

           surf_usm%rho_c_wall(nzb_wall,m)        = building_pars(ind_hc1_wall_r,building_type)
           surf_usm%rho_c_wall(nzb_wall+1,m)      = building_pars(ind_hc2_wall_r,building_type)
           surf_usm%rho_c_wall(nzb_wall+2,m)      = building_pars(ind_hc3_wall_r,building_type)
           surf_usm%rho_c_wall(nzb_wall+3,m)      = building_pars(ind_hc4_wall_r,building_type)
           surf_usm%lambda_h(nzb_wall,m)          = building_pars(ind_tc1_wall_r,building_type)
           surf_usm%lambda_h(nzb_wall+1,m)        = building_pars(ind_tc2_wall_r,building_type)
           surf_usm%lambda_h(nzb_wall+2,m)        = building_pars(ind_tc3_wall_r,building_type)
           surf_usm%lambda_h(nzb_wall+3,m)        = building_pars(ind_tc4_wall_r,building_type)
           surf_usm%rho_c_green(nzb_wall,m)       = rho_c_soil !building_pars(ind_hc1_wall_r,building_type)
           surf_usm%rho_c_green(nzb_wall+1,m)     = rho_c_soil !building_pars(ind_hc1_wall_r,building_type)
           surf_usm%rho_c_green(nzb_wall+2,m)     = rho_c_soil !building_pars(ind_hc2_wall_r,building_type)
           surf_usm%rho_c_green(nzb_wall+3,m)     = rho_c_soil !building_pars(ind_hc3_wall_r,building_type)
           surf_usm%lambda_h_green(nzb_wall,m)    = lambda_h_green_sm !building_pars(ind_tc1_wall_r,building_type)
           surf_usm%lambda_h_green(nzb_wall+1,m)  = lambda_h_green_sm !building_pars(ind_tc1_wall_r,building_type)
           surf_usm%lambda_h_green(nzb_wall+2,m)  = lambda_h_green_sm !building_pars(ind_tc2_wall_r,building_type)
           surf_usm%lambda_h_green(nzb_wall+3,m)  = lambda_h_green_sm !building_pars(ind_tc3_wall_r,building_type)
           surf_usm%rho_c_window(nzb_wall,m)      = building_pars(ind_hc1_win_r,building_type)
           surf_usm%rho_c_window(nzb_wall+1,m)    = building_pars(ind_hc2_win_r,building_type)
           surf_usm%rho_c_window(nzb_wall+2,m)    = building_pars(ind_hc3_win_r,building_type)
           surf_usm%rho_c_window(nzb_wall+3,m)    = building_pars(ind_hc4_win_r,building_type)
           surf_usm%lambda_h_window(nzb_wall,m)   = building_pars(ind_tc1_win_r,building_type)
           surf_usm%lambda_h_window(nzb_wall+1,m) = building_pars(ind_tc2_win_r,building_type)
           surf_usm%lambda_h_window(nzb_wall+2,m) = building_pars(ind_tc3_win_r,building_type)
           surf_usm%lambda_h_window(nzb_wall+3,m) = building_pars(ind_tc4_win_r,building_type)

           surf_usm%target_temp_summer(m)  = building_pars(ind_indoor_target_temp_summer,building_type)
           surf_usm%target_temp_winter(m)  = building_pars(ind_indoor_target_temp_winter,building_type)
!
!--        Emissivity of wall-, green- and window fraction
           surf_usm%emissivity(m,ind_veg_wall)  = building_pars(ind_emis_wall_r,building_type)
           surf_usm%emissivity(m,ind_pav_green) = building_pars(ind_emis_green_r,building_type)
           surf_usm%emissivity(m,ind_wat_win)   = building_pars(ind_emis_win_r,building_type)

           surf_usm%transmissivity(m)           = building_pars(ind_trans_r,building_type)

           surf_usm%z0(m)                       = building_pars(ind_z0,building_type)
           surf_usm%z0h(m)                      = building_pars(ind_z0qh,building_type)
           surf_usm%z0q(m)                      = building_pars(ind_z0qh,building_type)
!
!--        Albedo type for wall fraction, green fraction, window fraction
           surf_usm%albedo_type(m,ind_veg_wall)  = INT( building_pars(ind_alb_wall_r,building_type) )
           surf_usm%albedo_type(m,ind_pav_green) = INT( building_pars(ind_alb_green_r,building_type) )
           surf_usm%albedo_type(m,ind_wat_win)   = INT( building_pars(ind_alb_win_r,building_type) )

           surf_usm%zw(nzb_wall,m)         = building_pars(ind_thick_1_wall_r,building_type)
           surf_usm%zw(nzb_wall+1,m)       = building_pars(ind_thick_2_wall_r,building_type)
           surf_usm%zw(nzb_wall+2,m)       = building_pars(ind_thick_3_wall_r,building_type)
           surf_usm%zw(nzb_wall+3,m)       = building_pars(ind_thick_4_wall_r,building_type)

           surf_usm%zw_green(nzb_wall,m)         = building_pars(ind_thick_1_wall_r,building_type)
           surf_usm%zw_green(nzb_wall+1,m)       = building_pars(ind_thick_2_wall_r,building_type)
           surf_usm%zw_green(nzb_wall+2,m)       = building_pars(ind_thick_3_wall_r,building_type)
           surf_usm%zw_green(nzb_wall+3,m)       = building_pars(ind_thick_4_wall_r,building_type)

           surf_usm%zw_window(nzb_wall,m)         = building_pars(ind_thick_1_win_r,building_type)
           surf_usm%zw_window(nzb_wall+1,m)       = building_pars(ind_thick_2_win_r,building_type)
           surf_usm%zw_window(nzb_wall+2,m)       = building_pars(ind_thick_3_win_r,building_type)
           surf_usm%zw_window(nzb_wall+3,m)       = building_pars(ind_thick_4_win_r,building_type)

           surf_usm%green_type_roof(m)     = building_pars(ind_green_type_roof,building_type)
        ENDIF
     ENDDO
!
!--  Now, initialize facades (normal vector component and offset in z are zero).
     DO  m = 1, surf_usm%ns
        IF ( .NOT. ( surf_usm%upward(m)  .OR.  surf_usm%downward(m) ) )  THEN
           surf_usm%surface_types(m) = wall_category     !< Default category for root surface
!
!--        In order to distinguish between ground floor level and above-ground-floor level surfaces,
!--        set input indices.
           ind_alb_green    = MERGE( ind_alb_green_gfl,    ind_alb_green_agfl,    surf_usm%gfl(m) )
           ind_alb_wall     = MERGE( ind_alb_wall_gfl,     ind_alb_wall_agfl,     surf_usm%gfl(m) )
           ind_alb_win      = MERGE( ind_alb_win_gfl,      ind_alb_win_agfl,      surf_usm%gfl(m) )
           ind_wall_frac    = MERGE( ind_wall_frac_gfl,    ind_wall_frac_agfl,    surf_usm%gfl(m) )
           ind_win_frac     = MERGE( ind_win_frac_gfl,     ind_win_frac_agfl,     surf_usm%gfl(m) )
           ind_green_frac_w = MERGE( ind_green_frac_w_gfl, ind_green_frac_w_agfl, surf_usm%gfl(m) )
           ind_green_frac   = MERGE( ind_green_frac_r_gfl, ind_green_frac_r_agfl, surf_usm%gfl(m) )
           ind_lai          = MERGE( ind_lai_r_gfl,        ind_lai_r_agfl,        surf_usm%gfl(m) )
           ind_lai_w        = MERGE( ind_lai_w_gfl,        ind_lai_w_agfl,        surf_usm%gfl(m) )
           ind_hc1          = MERGE( ind_hc1_gfl,          ind_hc1_agfl,          surf_usm%gfl(m) )
           ind_hc1_win      = MERGE( ind_hc1_win_gfl,      ind_hc1_win_agfl,      surf_usm%gfl(m) )
           ind_hc2          = MERGE( ind_hc2_gfl,          ind_hc2_agfl,          surf_usm%gfl(m) )
           ind_hc2_win      = MERGE( ind_hc2_win_gfl,      ind_hc2_win_agfl,      surf_usm%gfl(m) )
           ind_hc3          = MERGE( ind_hc3_gfl,          ind_hc3_agfl,          surf_usm%gfl(m) )
           ind_hc3_win      = MERGE( ind_hc3_win_gfl,      ind_hc3_win_agfl,      surf_usm%gfl(m) )
           ind_hc4          = MERGE( ind_hc4_gfl,          ind_hc4_agfl,          surf_usm%gfl(m) )
           ind_hc4_win      = MERGE( ind_hc4_win_gfl,      ind_hc4_win_agfl,      surf_usm%gfl(m) )
           ind_tc1          = MERGE( ind_tc1_gfl,          ind_tc1_agfl,          surf_usm%gfl(m) )
           ind_tc1_win      = MERGE( ind_tc1_win_gfl,      ind_tc1_win_agfl,      surf_usm%gfl(m) )
           ind_tc2          = MERGE( ind_tc2_gfl,          ind_tc2_agfl,          surf_usm%gfl(m) )
           ind_tc2_win      = MERGE( ind_tc2_win_gfl,      ind_tc2_win_agfl,      surf_usm%gfl(m) )
           ind_tc3          = MERGE( ind_tc3_gfl,          ind_tc3_agfl,          surf_usm%gfl(m) )
           ind_tc3_win      = MERGE( ind_tc3_win_gfl,      ind_tc3_win_agfl,      surf_usm%gfl(m) )
           ind_tc4          = MERGE( ind_tc4_gfl,          ind_tc4_agfl,          surf_usm%gfl(m) )
           ind_tc4_win      = MERGE( ind_tc4_win_gfl,      ind_tc4_win_agfl,      surf_usm%gfl(m) )
           ind_thick_1      = MERGE( ind_thick_1_gfl,      ind_thick_1_agfl,      surf_usm%gfl(m) )
           ind_thick_1_win  = MERGE( ind_thick_1_win_gfl,  ind_thick_1_win_agfl,  surf_usm%gfl(m) )
           ind_thick_2      = MERGE( ind_thick_2_gfl,      ind_thick_2_agfl,      surf_usm%gfl(m) )
           ind_thick_2_win  = MERGE( ind_thick_2_win_gfl,  ind_thick_2_win_agfl,  surf_usm%gfl(m) )
           ind_thick_3      = MERGE( ind_thick_3_gfl,      ind_thick_3_agfl,      surf_usm%gfl(m) )
           ind_thick_3_win  = MERGE( ind_thick_3_win_gfl,  ind_thick_3_win_agfl,  surf_usm%gfl(m) )
           ind_thick_4      = MERGE( ind_thick_4_gfl,      ind_thick_4_agfl,      surf_usm%gfl(m) )
           ind_thick_4_win  = MERGE( ind_thick_4_win_gfl,  ind_thick_4_win_agfl,  surf_usm%gfl(m) )
           ind_emis_wall    = MERGE( ind_emis_wall_gfl,    ind_emis_wall_agfl,    surf_usm%gfl(m) )
           ind_emis_green   = MERGE( ind_emis_green_gfl,   ind_emis_green_agfl,   surf_usm%gfl(m) )
           ind_emis_win     = MERGE( ind_emis_win_gfl,     ind_emis_win_agfl,     surf_usm%gfl(m) )
           ind_trans        = MERGE( ind_trans_gfl,        ind_trans_agfl,        surf_usm%gfl(m) )
           ind_z0           = MERGE( ind_z0_gfl,           ind_z0_agfl,           surf_usm%gfl(m) )
           ind_z0qh         = MERGE( ind_z0qh_gfl,         ind_z0qh_agfl,         surf_usm%gfl(m) )
!
!--        Store building type and its name on each surface element
           surf_usm%building_type(m)      = building_type
           surf_usm%building_type_name(m) = building_type_name(building_type)
!
!--        Initialize relatvie wall- (0), green- (1) and window (2) fractions
           surf_usm%frac(m,ind_veg_wall)   = building_pars(ind_wall_frac,building_type)
           surf_usm%frac(m,ind_pav_green)  = building_pars(ind_green_frac_w,building_type)
           surf_usm%frac(m,ind_wat_win)    = building_pars(ind_win_frac,building_type)
           surf_usm%lai(m)                 = building_pars(ind_lai_w,building_type)

           surf_usm%rho_c_wall(nzb_wall,m)   = building_pars(ind_hc1,building_type)
           surf_usm%rho_c_wall(nzb_wall+1,m) = building_pars(ind_hc2,building_type)
           surf_usm%rho_c_wall(nzb_wall+2,m) = building_pars(ind_hc3,building_type)
           surf_usm%rho_c_wall(nzb_wall+3,m) = building_pars(ind_hc4,building_type)

           surf_usm%rho_c_green(nzb_wall,m)   = rho_c_soil !building_pars(ind_hc1,building_type)
           surf_usm%rho_c_green(nzb_wall+1,m) = rho_c_soil !building_pars(ind_hc1,building_type)
           surf_usm%rho_c_green(nzb_wall+2,m) = rho_c_soil !building_pars(ind_hc2,building_type)
           surf_usm%rho_c_green(nzb_wall+3,m) = rho_c_soil !building_pars(ind_hc3,building_type)

           surf_usm%rho_c_window(nzb_wall,m)   = building_pars(ind_hc1_win,building_type)
           surf_usm%rho_c_window(nzb_wall+1,m) = building_pars(ind_hc2_win,building_type)
           surf_usm%rho_c_window(nzb_wall+2,m) = building_pars(ind_hc3_win,building_type)
           surf_usm%rho_c_window(nzb_wall+3,m) = building_pars(ind_hc4_win,building_type)

           surf_usm%lambda_h(nzb_wall,m)   = building_pars(ind_tc1,building_type)
           surf_usm%lambda_h(nzb_wall+1,m) = building_pars(ind_tc2,building_type)
           surf_usm%lambda_h(nzb_wall+2,m) = building_pars(ind_tc3,building_type)
           surf_usm%lambda_h(nzb_wall+3,m) = building_pars(ind_tc4,building_type)

           surf_usm%lambda_h_green(nzb_wall,m)   = lambda_h_green_sm !building_pars(ind_tc1,building_type)
           surf_usm%lambda_h_green(nzb_wall+1,m) = lambda_h_green_sm !building_pars(ind_tc1,building_type)
           surf_usm%lambda_h_green(nzb_wall+2,m) = lambda_h_green_sm !building_pars(ind_tc2,building_type)
           surf_usm%lambda_h_green(nzb_wall+3,m) = lambda_h_green_sm !building_pars(ind_tc3,building_type)

           surf_usm%lambda_h_window(nzb_wall,m)   = building_pars(ind_tc1_win,building_type)
           surf_usm%lambda_h_window(nzb_wall+1,m) = building_pars(ind_tc2_win,building_type)
           surf_usm%lambda_h_window(nzb_wall+2,m) = building_pars(ind_tc3_win,building_type)
           surf_usm%lambda_h_window(nzb_wall+3,m) = building_pars(ind_tc4_win,building_type)

           surf_usm%target_temp_summer(m)  = building_pars(ind_indoor_target_temp_summer,building_type)
           surf_usm%target_temp_winter(m)  = building_pars(ind_indoor_target_temp_winter,building_type)
!
!--        Emissivity of wall-, green- and window fraction
           surf_usm%emissivity(m,ind_veg_wall)  = building_pars(ind_emis_wall,building_type)
           surf_usm%emissivity(m,ind_pav_green) = building_pars(ind_emis_green,building_type)
           surf_usm%emissivity(m,ind_wat_win)   = building_pars(ind_emis_win,building_type)

           surf_usm%transmissivity(m)      = building_pars(ind_trans,building_type)

           surf_usm%z0(m)                  = building_pars(ind_z0,building_type)
           surf_usm%z0h(m)                 = building_pars(ind_z0qh,building_type)
           surf_usm%z0q(m)                 = building_pars(ind_z0qh,building_type)

           surf_usm%albedo_type(m,ind_veg_wall)  = INT( building_pars(ind_alb_wall,building_type) )
           surf_usm%albedo_type(m,ind_pav_green) = INT( building_pars(ind_alb_green,building_type) )
           surf_usm%albedo_type(m,ind_wat_win)   = INT( building_pars(ind_alb_win,building_type) )

           surf_usm%zw(nzb_wall,m)         = building_pars(ind_thick_1,building_type)
           surf_usm%zw(nzb_wall+1,m)       = building_pars(ind_thick_2,building_type)
           surf_usm%zw(nzb_wall+2,m)       = building_pars(ind_thick_3,building_type)
           surf_usm%zw(nzb_wall+3,m)       = building_pars(ind_thick_4,building_type)

           surf_usm%zw_green(nzb_wall,m)         = building_pars(ind_thick_1,building_type)
           surf_usm%zw_green(nzb_wall+1,m)       = building_pars(ind_thick_2,building_type)
           surf_usm%zw_green(nzb_wall+2,m)       = building_pars(ind_thick_3,building_type)
           surf_usm%zw_green(nzb_wall+3,m)       = building_pars(ind_thick_4,building_type)

           surf_usm%zw_window(nzb_wall,m)        = building_pars(ind_thick_1_win,building_type)
           surf_usm%zw_window(nzb_wall+1,m)      = building_pars(ind_thick_2_win,building_type)
           surf_usm%zw_window(nzb_wall+2,m)      = building_pars(ind_thick_3_win,building_type)
           surf_usm%zw_window(nzb_wall+3,m)      = building_pars(ind_thick_4_win,building_type)

        ENDIF
     ENDDO
!
!--  Level 2 - initialization via building type read from file
     IF ( building_type_f%from_file )  THEN
        DO  m = 1, surf_usm%ns
           IF ( surf_usm%upward(m)  .OR.  surf_usm%downward(m) )  THEN
              i = surf_usm%i(m)
              j = surf_usm%j(m)
!
!--           For the moment, limit building type to 6 (to overcome errors in input file).
              st = building_type_f%var(j,i)
              IF ( st /= building_type_f%fill )  THEN

!
!--              In order to distinguish between ground floor level and above-ground-floor level
!--              surfaces, set input indices.
                 ind_green_frac = ind_green_frac_r_agfl
                 ind_lai        = ind_lai_r_agfl
                 ind_z0         = ind_z0_agfl
                 ind_z0qh       = ind_z0qh_agfl
!
!--              Store building type and its name on each surface element
                 surf_usm%building_type(m)      = st
                 surf_usm%building_type_name(m) = building_type_name(st)
!
!--              Initialize relatvie wall- (0), green- (1) and window (2) fractions
                 surf_usm%frac(m,ind_veg_wall)  = building_pars(ind_wall_frac_r,st)
                 surf_usm%frac(m,ind_pav_green) = building_pars(ind_green_frac,st)
                 surf_usm%frac(m,ind_wat_win)   = building_pars(ind_win_frac_r,st)
                 surf_usm%lai(m)                = building_pars(ind_lai,st)

                 surf_usm%rho_c_wall(nzb_wall,m)   = building_pars(ind_hc1_wall_r,st)
                 surf_usm%rho_c_wall(nzb_wall+1,m) = building_pars(ind_hc2_wall_r,st)
                 surf_usm%rho_c_wall(nzb_wall+2,m) = building_pars(ind_hc3_wall_r,st)
                 surf_usm%rho_c_wall(nzb_wall+3,m) = building_pars(ind_hc4_wall_r,st)
                 surf_usm%lambda_h(nzb_wall,m)     = building_pars(ind_tc1_wall_r,st)
                 surf_usm%lambda_h(nzb_wall+1,m)   = building_pars(ind_tc2_wall_r,st)
                 surf_usm%lambda_h(nzb_wall+2,m)   = building_pars(ind_tc3_wall_r,st)
                 surf_usm%lambda_h(nzb_wall+3,m)   = building_pars(ind_tc4_wall_r,st)

                 surf_usm%rho_c_green(nzb_wall,m)      = rho_c_soil !building_pars(ind_hc1_wall_r,st)
                 surf_usm%rho_c_green(nzb_wall+1,m)    = rho_c_soil !building_pars(ind_hc1_wall_r,st)
                 surf_usm%rho_c_green(nzb_wall+2,m)    = rho_c_soil !building_pars(ind_hc2_wall_r,st)
                 surf_usm%rho_c_green(nzb_wall+3,m)    = rho_c_soil !building_pars(ind_hc3_wall_r,st)
                 surf_usm%lambda_h_green(nzb_wall,m)   = lambda_h_green_sm !building_pars(ind_tc1_wall_r,st)
                 surf_usm%lambda_h_green(nzb_wall+1,m) = lambda_h_green_sm !building_pars(ind_tc1_wall_r,st)
                 surf_usm%lambda_h_green(nzb_wall+2,m) = lambda_h_green_sm !building_pars(ind_tc2_wall_r,st)
                 surf_usm%lambda_h_green(nzb_wall+3,m) = lambda_h_green_sm !building_pars(ind_tc3_wall_r,st)

                 surf_usm%rho_c_window(nzb_wall,m)      = building_pars(ind_hc1_win_r,st)
                 surf_usm%rho_c_window(nzb_wall+1,m)    = building_pars(ind_hc2_win_r,st)
                 surf_usm%rho_c_window(nzb_wall+2,m)    = building_pars(ind_hc3_win_r,st)
                 surf_usm%rho_c_window(nzb_wall+3,m)    = building_pars(ind_hc4_win_r,st)
                 surf_usm%lambda_h_window(nzb_wall,m)   = building_pars(ind_tc1_win_r,st)
                 surf_usm%lambda_h_window(nzb_wall+1,m) = building_pars(ind_tc2_win_r,st)
                 surf_usm%lambda_h_window(nzb_wall+2,m) = building_pars(ind_tc3_win_r,st)
                 surf_usm%lambda_h_window(nzb_wall+3,m) = building_pars(ind_tc4_win_r,st)

                 surf_usm%target_temp_summer(m)  = building_pars(ind_indoor_target_temp_summer,st)
                 surf_usm%target_temp_winter(m)  = building_pars(ind_indoor_target_temp_winter,st)
!
!--              Emissivity of wall-, green- and window fraction
                 surf_usm%emissivity(m,ind_veg_wall)  = building_pars(ind_emis_wall_r,st)
                 surf_usm%emissivity(m,ind_pav_green) = building_pars(ind_emis_green_r,st)
                 surf_usm%emissivity(m,ind_wat_win)   = building_pars(ind_emis_win_r,st)

                 surf_usm%transmissivity(m)      = building_pars(ind_trans_r,st)

                 surf_usm%z0(m)                  = building_pars(ind_z0,st)
                 surf_usm%z0h(m)                 = building_pars(ind_z0qh,st)
                 surf_usm%z0q(m)                 = building_pars(ind_z0qh,st)
!
!--              Albedo type for wall fraction, green fraction, window fraction
                 surf_usm%albedo_type(m,ind_veg_wall)  = INT( building_pars(ind_alb_wall_r,st) )
                 surf_usm%albedo_type(m,ind_pav_green) = INT( building_pars(ind_alb_green_r,st) )
                 surf_usm%albedo_type(m,ind_wat_win)   = INT( building_pars(ind_alb_win_r,st) )

                 surf_usm%zw(nzb_wall,m)   = building_pars(ind_thick_1_wall_r,st)
                 surf_usm%zw(nzb_wall+1,m) = building_pars(ind_thick_2_wall_r,st)
                 surf_usm%zw(nzb_wall+2,m) = building_pars(ind_thick_3_wall_r,st)
                 surf_usm%zw(nzb_wall+3,m) = building_pars(ind_thick_4_wall_r,st)

                 surf_usm%zw_green(nzb_wall,m)   = building_pars(ind_thick_1_wall_r,st)
                 surf_usm%zw_green(nzb_wall+1,m) = building_pars(ind_thick_2_wall_r,st)
                 surf_usm%zw_green(nzb_wall+2,m) = building_pars(ind_thick_3_wall_r,st)
                 surf_usm%zw_green(nzb_wall+3,m) = building_pars(ind_thick_4_wall_r,st)

                 surf_usm%zw_window(nzb_wall,m)   = building_pars(ind_thick_1_win_r,st)
                 surf_usm%zw_window(nzb_wall+1,m) = building_pars(ind_thick_2_win_r,st)
                 surf_usm%zw_window(nzb_wall+2,m) = building_pars(ind_thick_3_win_r,st)
                 surf_usm%zw_window(nzb_wall+3,m) = building_pars(ind_thick_4_win_r,st)

                 surf_usm%green_type_roof(m) = building_pars(ind_green_type_roof,st)

              ENDIF
           ENDIF
        ENDDO
!
!--     Now, initialize facades (normal vector component in z is zero).
        DO  m = 1, surf_usm%ns
           i = surf_usm%i(m) + surf_usm%ioff(m)
           j = surf_usm%j(m) + surf_usm%joff(m)
           st = building_type_f%var(j,i)

           IF ( .NOT. ( surf_usm%upward(m)  .OR.  surf_usm%downward(m) )  .AND.                    &
                st /= building_type_f%fill )  THEN
!
!--           In order to distinguish between ground floor level and above-ground-floor level
!--           surfaces, set input indices.
              ind_alb_green    = MERGE( ind_alb_green_gfl,    ind_alb_green_agfl,    surf_usm%gfl(m) )
              ind_alb_wall     = MERGE( ind_alb_wall_gfl,     ind_alb_wall_agfl,     surf_usm%gfl(m) )
              ind_alb_win      = MERGE( ind_alb_win_gfl,      ind_alb_win_agfl,      surf_usm%gfl(m) )
              ind_wall_frac    = MERGE( ind_wall_frac_gfl,    ind_wall_frac_agfl,    surf_usm%gfl(m) )
              ind_win_frac     = MERGE( ind_win_frac_gfl,     ind_win_frac_agfl,     surf_usm%gfl(m) )
              ind_green_frac_w = MERGE( ind_green_frac_w_gfl, ind_green_frac_w_agfl, surf_usm%gfl(m) )
              ind_lai_w        = MERGE( ind_lai_w_gfl,        ind_lai_w_agfl,        surf_usm%gfl(m) )
              ind_hc1          = MERGE( ind_hc1_gfl,          ind_hc1_agfl,          surf_usm%gfl(m) )
              ind_hc1_win      = MERGE( ind_hc1_win_gfl,      ind_hc1_win_agfl,      surf_usm%gfl(m) )
              ind_hc2          = MERGE( ind_hc2_gfl,          ind_hc2_agfl,          surf_usm%gfl(m) )
              ind_hc2_win      = MERGE( ind_hc2_win_gfl,      ind_hc2_win_agfl,      surf_usm%gfl(m) )
              ind_hc3          = MERGE( ind_hc3_gfl,          ind_hc3_agfl,          surf_usm%gfl(m) )
              ind_hc3_win      = MERGE( ind_hc3_win_gfl,      ind_hc3_win_agfl,      surf_usm%gfl(m) )
              ind_hc4          = MERGE( ind_hc4_gfl,          ind_hc4_agfl,          surf_usm%gfl(m) )
              ind_hc4_win      = MERGE( ind_hc4_win_gfl,      ind_hc4_win_agfl,      surf_usm%gfl(m) )
              ind_tc1          = MERGE( ind_tc1_gfl,          ind_tc1_agfl,          surf_usm%gfl(m) )
              ind_tc1_win      = MERGE( ind_tc1_win_gfl,      ind_tc1_win_agfl,      surf_usm%gfl(m) )
              ind_tc2          = MERGE( ind_tc2_gfl,          ind_tc2_agfl,          surf_usm%gfl(m) )
              ind_tc2_win      = MERGE( ind_tc2_win_gfl,      ind_tc2_win_agfl,      surf_usm%gfl(m) )
              ind_tc3          = MERGE( ind_tc3_gfl,          ind_tc3_agfl,          surf_usm%gfl(m) )
              ind_tc3_win      = MERGE( ind_tc3_win_gfl,      ind_tc3_win_agfl,      surf_usm%gfl(m) )
              ind_tc4          = MERGE( ind_tc4_gfl,          ind_tc4_agfl,          surf_usm%gfl(m) )
              ind_tc4_win      = MERGE( ind_tc4_win_gfl,      ind_tc4_win_agfl,      surf_usm%gfl(m) )
              ind_thick_1      = MERGE( ind_thick_1_gfl,      ind_thick_1_agfl,      surf_usm%gfl(m) )
              ind_thick_1_win  = MERGE( ind_thick_1_win_gfl,  ind_thick_1_win_agfl,  surf_usm%gfl(m) )
              ind_thick_2      = MERGE( ind_thick_2_gfl,      ind_thick_2_agfl,      surf_usm%gfl(m) )
              ind_thick_2_win  = MERGE( ind_thick_2_win_gfl,  ind_thick_2_win_agfl,  surf_usm%gfl(m) )
              ind_thick_3      = MERGE( ind_thick_3_gfl,      ind_thick_3_agfl,      surf_usm%gfl(m) )
              ind_thick_3_win  = MERGE( ind_thick_3_win_gfl,  ind_thick_3_win_agfl,  surf_usm%gfl(m) )
              ind_thick_4      = MERGE( ind_thick_4_gfl,      ind_thick_4_agfl,      surf_usm%gfl(m) )
              ind_thick_4_win  = MERGE( ind_thick_4_win_gfl,  ind_thick_4_win_agfl,  surf_usm%gfl(m) )
              ind_emis_wall    = MERGE( ind_emis_wall_gfl,    ind_emis_wall_agfl,    surf_usm%gfl(m) )
              ind_emis_green   = MERGE( ind_emis_green_gfl,   ind_emis_green_agfl,   surf_usm%gfl(m) )
              ind_emis_win     = MERGE( ind_emis_win_gfl,     ind_emis_win_agfl,     surf_usm%gfl(m) )
              ind_trans        = MERGE( ind_trans_gfl,        ind_trans_agfl,        surf_usm%gfl(m) )
              ind_z0           = MERGE( ind_z0_gfl,           ind_z0_agfl,           surf_usm%gfl(m) )
              ind_z0qh         = MERGE( ind_z0qh_gfl,         ind_z0qh_agfl,         surf_usm%gfl(m) )
!
!--           Store building type and its name on each surface element
              surf_usm%building_type(m)      = st
              surf_usm%building_type_name(m) = building_type_name(st)
!
!--           Initialize relatvie wall- (0), green- (1) and window (2) fractions
              surf_usm%frac(m,ind_veg_wall)  = building_pars(ind_wall_frac,st)
              surf_usm%frac(m,ind_pav_green) = building_pars(ind_green_frac_w,st)
              surf_usm%frac(m,ind_wat_win)   = building_pars(ind_win_frac,st)
              surf_usm%lai(m)                = building_pars(ind_lai_w,st)

              surf_usm%rho_c_wall(nzb_wall,m)   = building_pars(ind_hc1,st)
              surf_usm%rho_c_wall(nzb_wall+1,m) = building_pars(ind_hc2,st)
              surf_usm%rho_c_wall(nzb_wall+2,m) = building_pars(ind_hc3,st)
              surf_usm%rho_c_wall(nzb_wall+3,m) = building_pars(ind_hc4,st)

              surf_usm%rho_c_green(nzb_wall,m)   = rho_c_soil !building_pars(ind_hc1,st)
              surf_usm%rho_c_green(nzb_wall+1,m) = rho_c_soil !building_pars(ind_hc1,st)
              surf_usm%rho_c_green(nzb_wall+2,m) = rho_c_soil !building_pars(ind_hc2,st)
              surf_usm%rho_c_green(nzb_wall+3,m) = rho_c_soil !building_pars(ind_hc3,st)

              surf_usm%rho_c_window(nzb_wall,m)   = building_pars(ind_hc1_win,st)
              surf_usm%rho_c_window(nzb_wall+1,m) = building_pars(ind_hc2_win,st)
              surf_usm%rho_c_window(nzb_wall+2,m) = building_pars(ind_hc3_win,st)
              surf_usm%rho_c_window(nzb_wall+3,m) = building_pars(ind_hc4_win,st)

              surf_usm%lambda_h(nzb_wall,m)   = building_pars(ind_tc1,st)
              surf_usm%lambda_h(nzb_wall+1,m) = building_pars(ind_tc2,st)
              surf_usm%lambda_h(nzb_wall+2,m) = building_pars(ind_tc3,st)
              surf_usm%lambda_h(nzb_wall+3,m) = building_pars(ind_tc4,st)

              surf_usm%lambda_h_green(nzb_wall,m)   = lambda_h_green_sm !building_pars(ind_tc1,st)
              surf_usm%lambda_h_green(nzb_wall+1,m) = lambda_h_green_sm !building_pars(ind_tc1,st)
              surf_usm%lambda_h_green(nzb_wall+2,m) = lambda_h_green_sm !building_pars(ind_tc2,st)
              surf_usm%lambda_h_green(nzb_wall+3,m) = lambda_h_green_sm !building_pars(ind_tc3,st)

              surf_usm%lambda_h_window(nzb_wall,m)   = building_pars(ind_tc1_win,st)
              surf_usm%lambda_h_window(nzb_wall+1,m) = building_pars(ind_tc2_win,st)
              surf_usm%lambda_h_window(nzb_wall+2,m) = building_pars(ind_tc3_win,st)
              surf_usm%lambda_h_window(nzb_wall+3,m) = building_pars(ind_tc4_win,st)

              surf_usm%target_temp_summer(m) = building_pars(ind_indoor_target_temp_summer,st)
              surf_usm%target_temp_winter(m) = building_pars(ind_indoor_target_temp_winter,st)
!
!--           Emissivity of wall-, green- and window fraction
              surf_usm%emissivity(m,ind_veg_wall)  = building_pars(ind_emis_wall,st)
              surf_usm%emissivity(m,ind_pav_green) = building_pars(ind_emis_green,st)
              surf_usm%emissivity(m,ind_wat_win)   = building_pars(ind_emis_win,st)

              surf_usm%transmissivity(m) = building_pars(ind_trans,st)

              surf_usm%z0(m)  = building_pars(ind_z0,st)
              surf_usm%z0h(m) = building_pars(ind_z0qh,st)
              surf_usm%z0q(m) = building_pars(ind_z0qh,st)

              surf_usm%albedo_type(m,ind_veg_wall)  = INT( building_pars(ind_alb_wall,st) )
              surf_usm%albedo_type(m,ind_pav_green) = INT( building_pars(ind_alb_green,st) )
              surf_usm%albedo_type(m,ind_wat_win)   = INT( building_pars(ind_alb_win,st) )

              surf_usm%zw(nzb_wall,m)   = building_pars(ind_thick_1,st)
              surf_usm%zw(nzb_wall+1,m) = building_pars(ind_thick_2,st)
              surf_usm%zw(nzb_wall+2,m) = building_pars(ind_thick_3,st)
              surf_usm%zw(nzb_wall+3,m) = building_pars(ind_thick_4,st)

              surf_usm%zw_green(nzb_wall,m)   = building_pars(ind_thick_1,st)
              surf_usm%zw_green(nzb_wall+1,m) = building_pars(ind_thick_2,st)
              surf_usm%zw_green(nzb_wall+2,m) = building_pars(ind_thick_3,st)
              surf_usm%zw_green(nzb_wall+3,m) = building_pars(ind_thick_4,st)

              surf_usm%zw_window(nzb_wall,m)   = building_pars(ind_thick_1_win,st)
              surf_usm%zw_window(nzb_wall+1,m) = building_pars(ind_thick_2_win,st)
              surf_usm%zw_window(nzb_wall+2,m) = building_pars(ind_thick_3_win,st)
              surf_usm%zw_window(nzb_wall+3,m) = building_pars(ind_thick_4_win,st)
           ENDIF
        ENDDO
     ENDIF
!
!--  Level 3 - initialization via building_pars read from file. Note, only variables that are also
!--  defined in the input-standard can be initialized via file. Other variables will be initialized
!--  on level 1 or 2. Same as level 1 and 2, roofs and walls are initialized separately.
!--  !!! Note !!! The level 3 initialization is incomplete!
     IF ( building_pars_f%from_file )  THEN
        DO  m = 1, surf_usm%ns
           IF ( surf_usm%upward(m)  .OR.  surf_usm%downward(m) )  THEN
              i = surf_usm%i(m)
              j = surf_usm%j(m)

              ind_wall_frac    = ind_wall_frac_r
              ind_green_frac   = ind_green_frac_r_agfl
              ind_win_frac     = ind_win_frac_r
              ind_lai          = ind_lai_r_agfl
              ind_z0           = ind_z0_agfl
              ind_z0qh         = ind_z0qh_agfl
              ind_hc1          = ind_hc1_agfl
              ind_hc2          = ind_hc2_agfl
              ind_hc3          = ind_hc3_agfl
              ind_hc4          = ind_hc4_agfl
              ind_tc1          = ind_tc1_agfl
              ind_tc2          = ind_tc2_agfl
              ind_tc3          = ind_tc3_agfl
              ind_tc4          = ind_tc4_agfl
              ind_emis_wall    = ind_emis_wall_agfl
              ind_emis_green   = ind_emis_green_agfl
              ind_emis_win     = ind_emis_win_agfl
              ind_trans        = ind_trans_agfl
!
!--           Initialize relatvie wall- (0), green- (1) and window (2) fractions
              IF ( building_pars_f%pars_xy(ind_wall_frac,j,i) /= building_pars_f%fill )               &
                 surf_usm%frac(m,ind_veg_wall) = building_pars_f%pars_xy(ind_wall_frac,j,i)

              IF ( building_pars_f%pars_xy(ind_green_frac,j,i) /= building_pars_f%fill )              &
                 surf_usm%frac(m,ind_pav_green) = building_pars_f%pars_xy(ind_green_frac,j,i)

              IF ( building_pars_f%pars_xy(ind_win_frac,j,i) /= building_pars_f%fill )                &
                 surf_usm%frac(m,ind_wat_win) = building_pars_f%pars_xy(ind_win_frac,j,i)

              IF ( building_pars_f%pars_xy(ind_lai,j,i) /= building_pars_f%fill )                     &
                 surf_usm%lai(m) = building_pars_f%pars_xy(ind_lai,j,i)

              IF ( building_pars_f%pars_xy(ind_hc1,j,i) /= building_pars_f%fill )                     &
                 surf_usm%rho_c_wall(nzb_wall,m) = building_pars_f%pars_xy(ind_hc1,j,i)

              IF ( building_pars_f%pars_xy(ind_hc2,j,i) /= building_pars_f%fill )                     &
                 surf_usm%rho_c_wall(nzb_wall+1,m) = building_pars_f%pars_xy(ind_hc2,j,i)

              IF ( building_pars_f%pars_xy(ind_hc3,j,i) /= building_pars_f%fill )                     &
                 surf_usm%rho_c_wall(nzb_wall+2,m) = building_pars_f%pars_xy(ind_hc3,j,i)

              IF ( building_pars_f%pars_xy(ind_hc4,j,i) /= building_pars_f%fill )                     &
                 surf_usm%rho_c_wall(nzb_wall+3,m) = building_pars_f%pars_xy(ind_hc4,j,i)

              IF ( building_pars_f%pars_xy(ind_hc1,j,i) /= building_pars_f%fill )                     &
                 surf_usm%rho_c_green(nzb_wall,m) = building_pars_f%pars_xy(ind_hc1,j,i)

              IF ( building_pars_f%pars_xy(ind_hc2,j,i) /= building_pars_f%fill )                     &
                 surf_usm%rho_c_green(nzb_wall+1,m) = building_pars_f%pars_xy(ind_hc2,j,i)

              IF ( building_pars_f%pars_xy(ind_hc3,j,i) /= building_pars_f%fill )                     &
                 surf_usm%rho_c_green(nzb_wall+2,m) = building_pars_f%pars_xy(ind_hc3,j,i)

              IF ( building_pars_f%pars_xy(ind_hc4,j,i) /= building_pars_f%fill )                     &
                 surf_usm%rho_c_green(nzb_wall+3,m) = building_pars_f%pars_xy(ind_hc4,j,i)

              IF ( building_pars_f%pars_xy(ind_hc1,j,i) /= building_pars_f%fill )                     &
                 surf_usm%rho_c_window(nzb_wall,m) = building_pars_f%pars_xy(ind_hc1,j,i)

              IF ( building_pars_f%pars_xy(ind_hc2,j,i) /= building_pars_f%fill )                     &
                 surf_usm%rho_c_window(nzb_wall+1,m) = building_pars_f%pars_xy(ind_hc2,j,i)

              IF ( building_pars_f%pars_xy(ind_hc3,j,i) /= building_pars_f%fill )                     &
                 surf_usm%rho_c_window(nzb_wall+2,m) = building_pars_f%pars_xy(ind_hc3,j,i)

              IF ( building_pars_f%pars_xy(ind_hc4,j,i) /= building_pars_f%fill )                     &
                 surf_usm%rho_c_window(nzb_wall+3,m) = building_pars_f%pars_xy(ind_hc4,j,i)

              IF ( building_pars_f%pars_xy(ind_tc1,j,i) /= building_pars_f%fill )                     &
                 surf_usm%lambda_h(nzb_wall,m)   = building_pars_f%pars_xy(ind_tc1,j,i)

              IF ( building_pars_f%pars_xy(ind_tc2,j,i) /= building_pars_f%fill )                     &
                 surf_usm%lambda_h(nzb_wall+1,m) = building_pars_f%pars_xy(ind_tc2,j,i)

              IF ( building_pars_f%pars_xy(ind_tc3,j,i) /= building_pars_f%fill )                     &
                 surf_usm%lambda_h(nzb_wall+2,m) = building_pars_f%pars_xy(ind_tc3,j,i)

              IF ( building_pars_f%pars_xy(ind_tc4,j,i) /= building_pars_f%fill )                     &
                 surf_usm%lambda_h(nzb_wall+3,m) = building_pars_f%pars_xy(ind_tc4,j,i)

              IF ( building_pars_f%pars_xy(ind_tc1,j,i) /= building_pars_f%fill )                     &
                 surf_usm%lambda_h_green(nzb_wall,m) = building_pars_f%pars_xy(ind_tc1,j,i)

              IF ( building_pars_f%pars_xy(ind_tc2,j,i) /= building_pars_f%fill )                     &
                 surf_usm%lambda_h_green(nzb_wall+1,m) = building_pars_f%pars_xy(ind_tc2,j,i)

              IF ( building_pars_f%pars_xy(ind_tc3,j,i) /= building_pars_f%fill )                     &
                 surf_usm%lambda_h_green(nzb_wall+2,m) = building_pars_f%pars_xy(ind_tc3,j,i)

              IF ( building_pars_f%pars_xy(ind_tc4,j,i) /= building_pars_f%fill )                     &
                 surf_usm%lambda_h_green(nzb_wall+3,m) = building_pars_f%pars_xy(ind_tc4,j,i)

              IF ( building_pars_f%pars_xy(ind_tc1,j,i) /= building_pars_f%fill )                     &
                 surf_usm%lambda_h_window(nzb_wall,m) = building_pars_f%pars_xy(ind_tc1,j,i)

              IF ( building_pars_f%pars_xy(ind_tc2,j,i) /= building_pars_f%fill )                     &
                 surf_usm%lambda_h_window(nzb_wall+1,m) = building_pars_f%pars_xy(ind_tc2,j,i)

              IF ( building_pars_f%pars_xy(ind_tc3,j,i) /= building_pars_f%fill )                     &
                 surf_usm%lambda_h_window(nzb_wall+2,m) = building_pars_f%pars_xy(ind_tc3,j,i)

              IF ( building_pars_f%pars_xy(ind_tc4,j,i) /= building_pars_f%fill )                     &
                 surf_usm%lambda_h_window(nzb_wall+3,m) = building_pars_f%pars_xy(ind_tc4,j,i)

              IF ( building_pars_f%pars_xy(ind_indoor_target_temp_summer,j,i) /=                      &
                   building_pars_f%fill )                                                             &
                 surf_usm%target_temp_summer(m) = building_pars_f%pars_xy(ind_indoor_target_temp_summer,j,i)

              IF ( building_pars_f%pars_xy(ind_indoor_target_temp_winter,j,i) /=                      &
                   building_pars_f%fill )                                                             &
                 surf_usm%target_temp_winter(m) = building_pars_f%pars_xy(ind_indoor_target_temp_winter,j,i)

              IF ( building_pars_f%pars_xy(ind_emis_wall,j,i) /= building_pars_f%fill )               &
                 surf_usm%emissivity(m,ind_veg_wall) = building_pars_f%pars_xy(ind_emis_wall,j,i)

              IF ( building_pars_f%pars_xy(ind_emis_green,j,i) /= building_pars_f%fill )              &
                 surf_usm%emissivity(m,ind_pav_green) = building_pars_f%pars_xy(ind_emis_green,j,i)

              IF ( building_pars_f%pars_xy(ind_emis_win,j,i) /= building_pars_f%fill )                &
                 surf_usm%emissivity(m,ind_wat_win) = building_pars_f%pars_xy(ind_emis_win,j,i)

              IF ( building_pars_f%pars_xy(ind_trans,j,i) /= building_pars_f%fill )                   &
                 surf_usm%transmissivity(m) = building_pars_f%pars_xy(ind_trans,j,i)

              IF ( building_pars_f%pars_xy(ind_z0,j,i) /= building_pars_f%fill )                      &
                 surf_usm%z0(m) = building_pars_f%pars_xy(ind_z0,j,i)

              IF ( building_pars_f%pars_xy(ind_z0qh,j,i) /= building_pars_f%fill )                    &
                 surf_usm%z0h(m) = building_pars_f%pars_xy(ind_z0qh,j,i)

              IF ( building_pars_f%pars_xy(ind_z0qh,j,i) /= building_pars_f%fill )                    &
                 surf_usm%z0q(m) = building_pars_f%pars_xy(ind_z0qh,j,i)

              IF ( building_pars_f%pars_xy(ind_alb_wall_agfl,j,i) /= building_pars_f%fill )           &
                 surf_usm%albedo_type(m,ind_veg_wall)  = building_pars_f%pars_xy(ind_alb_wall_agfl,j,i)

              IF ( building_pars_f%pars_xy(ind_alb_green_agfl,j,i) /= building_pars_f%fill )          &
                 surf_usm%albedo_type(m,ind_pav_green) = building_pars_f%pars_xy(ind_alb_green_agfl,j,i)

              IF ( building_pars_f%pars_xy(ind_alb_win_agfl,j,i) /= building_pars_f%fill )            &
                 surf_usm%albedo_type(m,ind_wat_win) = building_pars_f%pars_xy(ind_alb_win_agfl,j,i)

              IF ( building_pars_f%pars_xy(ind_thick_1_agfl,j,i) /= building_pars_f%fill )            &
                 surf_usm%zw(nzb_wall,m) = building_pars_f%pars_xy(ind_thick_1_agfl,j,i)

              IF ( building_pars_f%pars_xy(ind_thick_2_agfl,j,i) /= building_pars_f%fill )            &
                 surf_usm%zw(nzb_wall+1,m) = building_pars_f%pars_xy(ind_thick_2_agfl,j,i)

              IF ( building_pars_f%pars_xy(ind_thick_3_agfl,j,i) /= building_pars_f%fill )            &
                 surf_usm%zw(nzb_wall+2,m) = building_pars_f%pars_xy(ind_thick_3_agfl,j,i)

              IF ( building_pars_f%pars_xy(ind_thick_4_agfl,j,i) /= building_pars_f%fill )            &
                 surf_usm%zw(nzb_wall+3,m) = building_pars_f%pars_xy(ind_thick_4_agfl,j,i)

              IF ( building_pars_f%pars_xy(ind_thick_1_agfl,j,i) /= building_pars_f%fill )            &
                 surf_usm%zw_green(nzb_wall,m) = building_pars_f%pars_xy(ind_thick_1_agfl,j,i)

              IF ( building_pars_f%pars_xy(ind_thick_2_agfl,j,i) /= building_pars_f%fill )            &
                 surf_usm%zw_green(nzb_wall+1,m) = building_pars_f%pars_xy(ind_thick_2_agfl,j,i)

              IF ( building_pars_f%pars_xy(ind_thick_3_agfl,j,i) /= building_pars_f%fill )            &
                 surf_usm%zw_green(nzb_wall+2,m) = building_pars_f%pars_xy(ind_thick_3_agfl,j,i)

              IF ( building_pars_f%pars_xy(ind_thick_4_agfl,j,i) /= building_pars_f%fill )            &
                 surf_usm%zw_green(nzb_wall+3,m) = building_pars_f%pars_xy(ind_thick_4_agfl,j,i)
           ENDIF
        ENDDO

        DO  m = 1, surf_usm%ns
           IF ( .NOT. ( surf_usm%upward(m)  .OR.  surf_usm%downward(m) ) )  THEN
              i = surf_usm%i(m) + surf_usm%ioff(m)
              j = surf_usm%j(m) + surf_usm%joff(m)
!
!--           In order to distinguish between ground floor level and above-ground-floor level
!--           surfaces, set input indices.
              ind_wall_frac    = MERGE( ind_wall_frac_gfl,    ind_wall_frac_agfl,    surf_usm%gfl(m) )
              ind_green_frac_w = MERGE( ind_green_frac_w_gfl, ind_green_frac_w_agfl, surf_usm%gfl(m) )
              ind_win_frac     = MERGE( ind_win_frac_gfl,     ind_win_frac_agfl,     surf_usm%gfl(m) )
              ind_lai_w        = MERGE( ind_lai_w_gfl,        ind_lai_w_agfl,        surf_usm%gfl(m) )
              ind_z0           = MERGE( ind_z0_gfl,           ind_z0_agfl,           surf_usm%gfl(m) )
              ind_z0qh         = MERGE( ind_z0qh_gfl,         ind_z0qh_agfl,         surf_usm%gfl(m) )
              ind_hc1          = MERGE( ind_hc1_gfl,          ind_hc1_agfl,          surf_usm%gfl(m) )
              ind_hc2          = MERGE( ind_hc2_gfl,          ind_hc2_agfl,          surf_usm%gfl(m) )
              ind_hc3          = MERGE( ind_hc3_gfl,          ind_hc3_agfl,          surf_usm%gfl(m) )
              ind_hc4          = MERGE( ind_hc4_gfl,          ind_hc4_agfl,          surf_usm%gfl(m) )
              ind_tc1          = MERGE( ind_tc1_gfl,          ind_tc1_agfl,          surf_usm%gfl(m) )
              ind_tc2          = MERGE( ind_tc2_gfl,          ind_tc2_agfl,          surf_usm%gfl(m) )
              ind_tc3          = MERGE( ind_tc3_gfl,          ind_tc3_agfl,          surf_usm%gfl(m) )
              ind_tc4          = MERGE( ind_tc4_gfl,          ind_tc4_agfl,          surf_usm%gfl(m) )
              ind_emis_wall    = MERGE( ind_emis_wall_gfl,    ind_emis_wall_agfl,    surf_usm%gfl(m) )
              ind_emis_green   = MERGE( ind_emis_green_gfl,   ind_emis_green_agfl,   surf_usm%gfl(m) )
              ind_emis_win     = MERGE( ind_emis_win_gfl,     ind_emis_win_agfl,     surf_usm%gfl(m) )
              ind_trans        = MERGE( ind_trans_gfl,        ind_trans_agfl,        surf_usm%gfl(m) )
!
!--           Initialize relatvie wall- (0), green- (1) and window (2) fractions
              IF ( building_pars_f%pars_xy(ind_wall_frac,j,i) /= building_pars_f%fill )            &
                 surf_usm%frac(m,ind_veg_wall) = building_pars_f%pars_xy(ind_wall_frac,j,i)

              IF ( building_pars_f%pars_xy(ind_green_frac_w,j,i) /= building_pars_f%fill )         &
                 surf_usm%frac(m,ind_pav_green) = building_pars_f%pars_xy(ind_green_frac_w,j,i)

              IF ( building_pars_f%pars_xy(ind_win_frac,j,i) /= building_pars_f%fill )             &
                 surf_usm%frac(m,ind_wat_win) = building_pars_f%pars_xy(ind_win_frac,j,i)

              IF ( building_pars_f%pars_xy(ind_lai_w,j,i) /= building_pars_f%fill )                &
                 surf_usm%lai(m) = building_pars_f%pars_xy(ind_lai_w,j,i)

              IF ( building_pars_f%pars_xy(ind_hc1,j,i) /= building_pars_f%fill )                  &
                 surf_usm%rho_c_wall(nzb_wall,m) = building_pars_f%pars_xy(ind_hc1,j,i)

              IF ( building_pars_f%pars_xy(ind_hc2,j,i) /= building_pars_f%fill )                  &
                 surf_usm%rho_c_wall(nzb_wall+1,m) = building_pars_f%pars_xy(ind_hc2,j,i)

              IF ( building_pars_f%pars_xy(ind_hc3,j,i) /= building_pars_f%fill )                  &
                 surf_usm%rho_c_wall(nzb_wall+2,m) = building_pars_f%pars_xy(ind_hc3,j,i)

              IF ( building_pars_f%pars_xy(ind_hc4,j,i) /= building_pars_f%fill )                  &
                 surf_usm%rho_c_wall(nzb_wall+3,m) = building_pars_f%pars_xy(ind_hc4,j,i)

              IF ( building_pars_f%pars_xy(ind_hc1,j,i) /= building_pars_f%fill )                  &
                 surf_usm%rho_c_green(nzb_wall,m) = building_pars_f%pars_xy(ind_hc1,j,i)

              IF ( building_pars_f%pars_xy(ind_hc2,j,i) /= building_pars_f%fill )                  &
                 surf_usm%rho_c_green(nzb_wall+1,m) = building_pars_f%pars_xy(ind_hc2,j,i)

              IF ( building_pars_f%pars_xy(ind_hc3,j,i) /= building_pars_f%fill )                  &
                 surf_usm%rho_c_green(nzb_wall+2,m) = building_pars_f%pars_xy(ind_hc3,j,i)

              IF ( building_pars_f%pars_xy(ind_hc4,j,i) /= building_pars_f%fill )                  &
                 surf_usm%rho_c_green(nzb_wall+3,m) = building_pars_f%pars_xy(ind_hc4,j,i)

              IF ( building_pars_f%pars_xy(ind_hc1,j,i) /= building_pars_f%fill )                  &
                 surf_usm%rho_c_window(nzb_wall,m) = building_pars_f%pars_xy(ind_hc1,j,i)

              IF ( building_pars_f%pars_xy(ind_hc2,j,i) /= building_pars_f%fill )                  &
                 surf_usm%rho_c_window(nzb_wall+1,m) = building_pars_f%pars_xy(ind_hc2,j,i)

              IF ( building_pars_f%pars_xy(ind_hc3,j,i) /= building_pars_f%fill )                  &
                 surf_usm%rho_c_window(nzb_wall+2,m) = building_pars_f%pars_xy(ind_hc3,j,i)

              IF ( building_pars_f%pars_xy(ind_hc4,j,i) /= building_pars_f%fill )                  &
                 surf_usm%rho_c_window(nzb_wall+3,m) = building_pars_f%pars_xy(ind_hc4,j,i)

              IF ( building_pars_f%pars_xy(ind_tc1,j,i) /= building_pars_f%fill )                  &
                 surf_usm%lambda_h(nzb_wall,m) = building_pars_f%pars_xy(ind_tc1,j,i)

              IF ( building_pars_f%pars_xy(ind_tc2,j,i) /= building_pars_f%fill )                  &
                 surf_usm%lambda_h(nzb_wall+1,m) = building_pars_f%pars_xy(ind_tc2,j,i)

              IF ( building_pars_f%pars_xy(ind_tc3,j,i) /= building_pars_f%fill )                  &
                 surf_usm%lambda_h(nzb_wall+2,m) = building_pars_f%pars_xy(ind_tc3,j,i)

              IF ( building_pars_f%pars_xy(ind_tc4,j,i) /= building_pars_f%fill )                  &
                 surf_usm%lambda_h(nzb_wall+3,m) = building_pars_f%pars_xy(ind_tc4,j,i)

              IF ( building_pars_f%pars_xy(ind_tc1,j,i) /= building_pars_f%fill )                  &
                 surf_usm%lambda_h_green(nzb_wall,m) = building_pars_f%pars_xy(ind_tc1,j,i)

              IF ( building_pars_f%pars_xy(ind_tc2,j,i) /= building_pars_f%fill )                  &
                 surf_usm%lambda_h_green(nzb_wall+1,m) = building_pars_f%pars_xy(ind_tc2,j,i)

              IF ( building_pars_f%pars_xy(ind_tc3,j,i) /= building_pars_f%fill )                  &
                 surf_usm%lambda_h_green(nzb_wall+2,m) = building_pars_f%pars_xy(ind_tc3,j,i)

              IF ( building_pars_f%pars_xy(ind_tc4,j,i) /= building_pars_f%fill )                  &
                 surf_usm%lambda_h_green(nzb_wall+3,m) = building_pars_f%pars_xy(ind_tc4,j,i)

              IF ( building_pars_f%pars_xy(ind_tc1,j,i) /= building_pars_f%fill )                  &
                 surf_usm%lambda_h_window(nzb_wall,m) = building_pars_f%pars_xy(ind_tc1,j,i)

              IF ( building_pars_f%pars_xy(ind_tc2,j,i) /= building_pars_f%fill )                  &
                 surf_usm%lambda_h_window(nzb_wall+1,m) = building_pars_f%pars_xy(ind_tc2,j,i)

              IF ( building_pars_f%pars_xy(ind_tc3,j,i) /= building_pars_f%fill )                  &
                 surf_usm%lambda_h_window(nzb_wall+2,m) = building_pars_f%pars_xy(ind_tc3,j,i)

              IF ( building_pars_f%pars_xy(ind_tc4,j,i) /= building_pars_f%fill )                  &
                 surf_usm%lambda_h_window(nzb_wall+3,m) = building_pars_f%pars_xy(ind_tc4,j,i)

              IF ( building_pars_f%pars_xy(ind_indoor_target_temp_summer,j,i) /= building_pars_f%fill ) &
                 surf_usm%target_temp_summer(m) = building_pars_f%pars_xy(ind_indoor_target_temp_summer,j,i)

              IF ( building_pars_f%pars_xy(ind_indoor_target_temp_winter,j,i) /= building_pars_f%fill ) &
                 surf_usm%target_temp_winter(m) = building_pars_f%pars_xy(ind_indoor_target_temp_winter,j,i)

              IF ( building_pars_f%pars_xy(ind_emis_wall,j,i) /= building_pars_f%fill )            &
                 surf_usm%emissivity(m,ind_veg_wall) = building_pars_f%pars_xy(ind_emis_wall,j,i)

              IF ( building_pars_f%pars_xy(ind_emis_green,j,i) /= building_pars_f%fill )           &
                 surf_usm%emissivity(m,ind_pav_green) = building_pars_f%pars_xy(ind_emis_green,j,i)

              IF ( building_pars_f%pars_xy(ind_emis_win,j,i) /= building_pars_f%fill )             &
                 surf_usm%emissivity(m,ind_wat_win) = building_pars_f%pars_xy(ind_emis_win,j,i)

              IF ( building_pars_f%pars_xy(ind_trans,j,i) /= building_pars_f%fill )                &
                 surf_usm%transmissivity(m) = building_pars_f%pars_xy(ind_trans,j,i)

              IF ( building_pars_f%pars_xy(ind_z0,j,i) /= building_pars_f%fill )                   &
                 surf_usm%z0(m) = building_pars_f%pars_xy(ind_z0,j,i)

              IF ( building_pars_f%pars_xy(ind_z0qh,j,i) /= building_pars_f%fill )                 &
                 surf_usm%z0h(m) = building_pars_f%pars_xy(ind_z0qh,j,i)

              IF ( building_pars_f%pars_xy(ind_z0qh,j,i) /= building_pars_f%fill )                 &
                 surf_usm%z0q(m) = building_pars_f%pars_xy(ind_z0qh,j,i)

              IF ( building_pars_f%pars_xy(ind_alb_wall_agfl,j,i) /= building_pars_f%fill )        &
                 surf_usm%albedo_type(m,ind_veg_wall)  = building_pars_f%pars_xy(ind_alb_wall_agfl,j,i)

              IF ( building_pars_f%pars_xy(ind_alb_green_agfl,j,i) /= building_pars_f%fill )       &
                 surf_usm%albedo_type(m,ind_pav_green) = building_pars_f%pars_xy(ind_alb_green_agfl,j,i)

              IF ( building_pars_f%pars_xy(ind_alb_win_agfl,j,i) /= building_pars_f%fill )         &
                 surf_usm%albedo_type(m,ind_wat_win) = building_pars_f%pars_xy(ind_alb_win_agfl,j,i)

              IF ( building_pars_f%pars_xy(ind_thick_1_agfl,j,i) /= building_pars_f%fill )         &
                 surf_usm%zw(nzb_wall,m) = building_pars_f%pars_xy(ind_thick_1_agfl,j,i)

              IF ( building_pars_f%pars_xy(ind_thick_2_agfl,j,i) /= building_pars_f%fill )         &
                 surf_usm%zw(nzb_wall+1,m) = building_pars_f%pars_xy(ind_thick_2_agfl,j,i)

              IF ( building_pars_f%pars_xy(ind_thick_3_agfl,j,i) /= building_pars_f%fill )         &
                 surf_usm%zw(nzb_wall+2,m) = building_pars_f%pars_xy(ind_thick_3_agfl,j,i)

              IF ( building_pars_f%pars_xy(ind_thick_4_agfl,j,i) /= building_pars_f%fill )         &
                 surf_usm%zw(nzb_wall+3,m) = building_pars_f%pars_xy(ind_thick_4_agfl,j,i)

              IF ( building_pars_f%pars_xy(ind_thick_1_agfl,j,i) /= building_pars_f%fill )         &
                 surf_usm%zw_green(nzb_wall,m) = building_pars_f%pars_xy(ind_thick_1_agfl,j,i)

              IF ( building_pars_f%pars_xy(ind_thick_2_agfl,j,i) /= building_pars_f%fill )         &
                 surf_usm%zw_green(nzb_wall+1,m) = building_pars_f%pars_xy(ind_thick_2_agfl,j,i)

              IF ( building_pars_f%pars_xy(ind_thick_3_agfl,j,i) /= building_pars_f%fill )         &
                 surf_usm%zw_green(nzb_wall+2,m) = building_pars_f%pars_xy(ind_thick_3_agfl,j,i)

              IF ( building_pars_f%pars_xy(ind_thick_4_agfl,j,i) /= building_pars_f%fill )         &
                 surf_usm%zw_green(nzb_wall+3,m) = building_pars_f%pars_xy(ind_thick_4_agfl,j,i)

           ENDIF
        ENDDO
     ENDIF
!
!--  Read building surface pars. If present, they override LOD1-LOD3 building pars where applicable.
!--  Same as LOD1-LOD3 initialization, roofs and walls are initialized separately.
     IF ( building_surface_pars_f%from_file )  THEN
        DO  m = 1, surf_usm%ns
           i = surf_usm%i(m)
           j = surf_usm%j(m)
           k = surf_usm%k(m)
!
!--        Iterate over surfaces in column, check height and orientation
           DO  is = building_surface_pars_f%index_ji(1,j,i), building_surface_pars_f%index_ji(2,j,i)
              IF ( building_surface_pars_f%coords(4,is) == -surf_usm%koff(m) .AND.                 &
                   building_surface_pars_f%coords(1,is) == k )                                     &
              THEN

                 IF ( building_surface_pars_f%pars(ind_s_wall_frac,is) /=                          &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%frac(m,ind_veg_wall) =                                                &
                                building_surface_pars_f%pars(ind_s_wall_frac,is)

                 IF ( building_surface_pars_f%pars(ind_s_green_frac_w,is) /=                       &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%frac(m,ind_pav_green) =                                               &
                                building_surface_pars_f%pars(ind_s_green_frac_w,is)

                 IF ( building_surface_pars_f%pars(ind_s_green_frac_r,is) /=                       &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%frac(m,ind_pav_green) =                                               &
                                building_surface_pars_f%pars(ind_s_green_frac_r,is)
                    !TODO clarify: why should _w and _r be on the same surface?

                 IF ( building_surface_pars_f%pars(ind_s_win_frac,is) /=                           &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%frac(m,ind_wat_win) = building_surface_pars_f%pars(ind_s_win_frac,is)

                 IF ( building_surface_pars_f%pars(ind_s_lai_r,is) /=                              &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%lai(m) = building_surface_pars_f%pars(ind_s_lai_r,is)

                 IF ( building_surface_pars_f%pars(ind_s_hc1,is) /=                                &
                      building_surface_pars_f%fill )                                               &
                 THEN
                    surf_usm%rho_c_wall(nzb_wall:nzb_wall+1,m) =                                   &
                                building_surface_pars_f%pars(ind_s_hc1,is)
                    surf_usm%rho_c_green(nzb_wall:nzb_wall+1,m) =                                  &
                                building_surface_pars_f%pars(ind_s_hc1,is)
                    surf_usm%rho_c_window(nzb_wall:nzb_wall+1,m) =                                 &
                                building_surface_pars_f%pars(ind_s_hc1,is)
                 ENDIF

                 IF ( building_surface_pars_f%pars(ind_s_hc2,is) /=                                &
                      building_surface_pars_f%fill )                                               &
                 THEN
                    surf_usm%rho_c_wall(nzb_wall+2,m) =                                            &
                                building_surface_pars_f%pars(ind_s_hc2,is)
                    surf_usm%rho_c_green(nzb_wall+2,m) =                                           &
                                building_surface_pars_f%pars(ind_s_hc2,is)
                    surf_usm%rho_c_window(nzb_wall+2,m) =                                          &
                                building_surface_pars_f%pars(ind_s_hc2,is)
                 ENDIF

                 IF ( building_surface_pars_f%pars(ind_s_hc3,is) /=                                &
                      building_surface_pars_f%fill )                                               &
                 THEN
                    surf_usm%rho_c_wall(nzb_wall+3,m) =                                            &
                                building_surface_pars_f%pars(ind_s_hc3,is)
                    surf_usm%rho_c_green(nzb_wall+3,m) =                                           &
                                building_surface_pars_f%pars(ind_s_hc3,is)
                    surf_usm%rho_c_window(nzb_wall+3,m) =                                          &
                                building_surface_pars_f%pars(ind_s_hc3,is)
                 ENDIF

                 IF ( building_surface_pars_f%pars(ind_s_tc1,is) /=                                &
                      building_surface_pars_f%fill )                                               &
                 THEN
                    surf_usm%lambda_h(nzb_wall:nzb_wall+1,m) =                                     &
                                building_surface_pars_f%pars(ind_s_tc1,is)
                    surf_usm%lambda_h_green(nzb_wall:nzb_wall+1,m) =                               &
                                building_surface_pars_f%pars(ind_s_tc1,is)
                    surf_usm%lambda_h_window(nzb_wall:nzb_wall+1,m) =                              &
                                building_surface_pars_f%pars(ind_s_tc1,is)
                 ENDIF

                 IF ( building_surface_pars_f%pars(ind_s_tc2,is) /=                                &
                      building_surface_pars_f%fill )                                               &
                 THEN
                    surf_usm%lambda_h(nzb_wall+2,m) =                                              &
                                building_surface_pars_f%pars(ind_s_tc2,is)
                    surf_usm%lambda_h_green(nzb_wall+2,m) =                                        &
                                building_surface_pars_f%pars(ind_s_tc2,is)
                    surf_usm%lambda_h_window(nzb_wall+2,m) =                                       &
                                building_surface_pars_f%pars(ind_s_tc2,is)
                 ENDIF

                 IF ( building_surface_pars_f%pars(ind_s_tc3,is) /=                                &
                      building_surface_pars_f%fill )                                               &
                 THEN
                    surf_usm%lambda_h(nzb_wall+3,m) =                                              &
                                building_surface_pars_f%pars(ind_s_tc3,is)
                    surf_usm%lambda_h_green(nzb_wall+3,m) =                                        &
                                building_surface_pars_f%pars(ind_s_tc3,is)
                    surf_usm%lambda_h_window(nzb_wall+3,m) =                                       &
                                building_surface_pars_f%pars(ind_s_tc3,is)
                 ENDIF

                 IF ( building_surface_pars_f%pars(ind_s_indoor_target_temp_summer,is) /=          &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%target_temp_summer(m) =                                               &
                                building_surface_pars_f%pars(ind_s_indoor_target_temp_summer,is)

                 IF ( building_surface_pars_f%pars(ind_s_indoor_target_temp_winter,is) /=          &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%target_temp_winter(m) =                                               &
                                building_surface_pars_f%pars(ind_s_indoor_target_temp_winter,is)

                 IF ( building_surface_pars_f%pars(ind_s_emis_wall,is) /=                          &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%emissivity(m,ind_veg_wall) =                                          &
                                building_surface_pars_f%pars(ind_s_emis_wall,is)

                 IF ( building_surface_pars_f%pars(ind_s_emis_green,is) /=                         &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%emissivity(m,ind_pav_green) =                                         &
                                building_surface_pars_f%pars(ind_s_emis_green,is)

                 IF ( building_surface_pars_f%pars(ind_s_emis_win,is) /=                           &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%emissivity(m,ind_wat_win) =                                           &
                                building_surface_pars_f%pars(ind_s_emis_win,is)

                 IF ( building_surface_pars_f%pars(ind_s_trans,is) /=                              &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%transmissivity(m) = building_surface_pars_f%pars(ind_s_trans,is)

                 IF ( building_surface_pars_f%pars(ind_s_z0,is) /=                                 &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%z0(m) = building_surface_pars_f%pars(ind_s_z0,is)

                 IF ( building_surface_pars_f%pars(ind_s_z0qh,is) /=                               &
                      building_surface_pars_f%fill )                                               &
                 THEN
                    surf_usm%z0q(m) = building_surface_pars_f%pars(ind_s_z0qh,is)
                    surf_usm%z0h(m) = building_surface_pars_f%pars(ind_s_z0qh,is)
                 ENDIF

                 EXIT ! Surface was found and processed
              ENDIF
           ENDDO
        ENDDO

        DO  m = 1, surf_usm%ns
           i = surf_usm%i(m)
           j = surf_usm%j(m)
           k = surf_usm%k(m)
!
!--        Iterate over surfaces in column, check height and orientation
           DO  is = building_surface_pars_f%index_ji(1,j,i), building_surface_pars_f%index_ji(2,j,i)
              IF ( building_surface_pars_f%coords(5,is) == -surf_usm%joff(m) .AND.                 &
                   building_surface_pars_f%coords(6,is) == -surf_usm%ioff(m) .AND.                 &
                   building_surface_pars_f%coords(1,is) == k )                                     &
              THEN

                 IF ( building_surface_pars_f%pars(ind_s_wall_frac,is) /=                          &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%frac(m,ind_veg_wall) =                                                &
                              building_surface_pars_f%pars(ind_s_wall_frac,is)

                 IF ( building_surface_pars_f%pars(ind_s_green_frac_w,is) /=                       &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%frac(m,ind_pav_green) =                                               &
                              building_surface_pars_f%pars(ind_s_green_frac_w,is)

                 IF ( building_surface_pars_f%pars(ind_s_green_frac_r,is) /=                       &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%frac(m,ind_pav_green) =                                               &
                              building_surface_pars_f%pars(ind_s_green_frac_r,is)
                    !TODO Clarify: why should _w and _r be on the same surface?

                 IF ( building_surface_pars_f%pars(ind_s_win_frac,is) /=                           &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%frac(m,ind_wat_win) =                                                 &
                              building_surface_pars_f%pars(ind_s_win_frac,is)

                 IF ( building_surface_pars_f%pars(ind_s_lai_r,is) /=                              &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%lai(m) = building_surface_pars_f%pars(ind_s_lai_r,is)

                 IF ( building_surface_pars_f%pars(ind_s_hc1,is) /=                                &
                      building_surface_pars_f%fill )                                               &
                 THEN
                    surf_usm%rho_c_wall(nzb_wall:nzb_wall+1,m) =                                   &
                              building_surface_pars_f%pars(ind_s_hc1,is)
                    surf_usm%rho_c_green(nzb_wall:nzb_wall+1,m) =                                  &
                              building_surface_pars_f%pars(ind_s_hc1,is)
                    surf_usm%rho_c_window(nzb_wall:nzb_wall+1,m) =                                 &
                              building_surface_pars_f%pars(ind_s_hc1,is)
                 ENDIF

                 IF ( building_surface_pars_f%pars(ind_s_hc2,is) /=                                &
                      building_surface_pars_f%fill )                                               &
                 THEN
                    surf_usm%rho_c_wall(nzb_wall+2,m) =                                            &
                              building_surface_pars_f%pars(ind_s_hc2,is)
                    surf_usm%rho_c_green(nzb_wall+2,m) =                                           &
                              building_surface_pars_f%pars(ind_s_hc2,is)
                    surf_usm%rho_c_window(nzb_wall+2,m) =                                          &
                              building_surface_pars_f%pars(ind_s_hc2,is)
                 ENDIF

                 IF ( building_surface_pars_f%pars(ind_s_hc3,is) /=                                &
                      building_surface_pars_f%fill )                                               &
                 THEN
                    surf_usm%rho_c_wall(nzb_wall+3,m) =                                            &
                              building_surface_pars_f%pars(ind_s_hc3,is)
                    surf_usm%rho_c_green(nzb_wall+3,m) =                                           &
                              building_surface_pars_f%pars(ind_s_hc3,is)
                    surf_usm%rho_c_window(nzb_wall+3,m) =                                          &
                              building_surface_pars_f%pars(ind_s_hc3,is)
                 ENDIF

                 IF ( building_surface_pars_f%pars(ind_s_tc1,is) /=                                &
                      building_surface_pars_f%fill )                                               &
                 THEN
                    surf_usm%lambda_h(nzb_wall:nzb_wall+1,m) =                                     &
                              building_surface_pars_f%pars(ind_s_tc1,is)
                    surf_usm%lambda_h_green(nzb_wall:nzb_wall+1,m) =                               &
                              building_surface_pars_f%pars(ind_s_tc1,is)
                    surf_usm%lambda_h_window(nzb_wall:nzb_wall+1,m) =                              &
                              building_surface_pars_f%pars(ind_s_tc1,is)
                 ENDIF

                 IF ( building_surface_pars_f%pars(ind_s_tc2,is) /=                                &
                      building_surface_pars_f%fill )                                               &
                 THEN
                    surf_usm%lambda_h(nzb_wall+2,m) =                                              &
                              building_surface_pars_f%pars(ind_s_tc2,is)
                    surf_usm%lambda_h_green(nzb_wall+2,m) =                                        &
                              building_surface_pars_f%pars(ind_s_tc2,is)
                    surf_usm%lambda_h_window(nzb_wall+2,m) =                                       &
                              building_surface_pars_f%pars(ind_s_tc2,is)
                 ENDIF

                 IF ( building_surface_pars_f%pars(ind_s_tc3,is) /=                                &
                      building_surface_pars_f%fill )                                               &
                 THEN
                    surf_usm%lambda_h(nzb_wall+3,m) =                                              &
                              building_surface_pars_f%pars(ind_s_tc3,is)
                    surf_usm%lambda_h_green(nzb_wall+3,m) =                                        &
                              building_surface_pars_f%pars(ind_s_tc3,is)
                    surf_usm%lambda_h_window(nzb_wall+3,m) =                                       &
                              building_surface_pars_f%pars(ind_s_tc3,is)
                 ENDIF

                 IF ( building_surface_pars_f%pars(ind_s_indoor_target_temp_summer,is) /=          &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%target_temp_summer(m) =                                               &
                              building_surface_pars_f%pars(ind_s_indoor_target_temp_summer,is)

                 IF ( building_surface_pars_f%pars(ind_s_indoor_target_temp_winter,is) /=          &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%target_temp_winter(m) =                                               &
                              building_surface_pars_f%pars(ind_s_indoor_target_temp_winter,is)

                 IF ( building_surface_pars_f%pars(ind_s_emis_wall,is) /=                          &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%emissivity(m,ind_veg_wall) =                                          &
                              building_surface_pars_f%pars(ind_s_emis_wall,is)

                 IF ( building_surface_pars_f%pars(ind_s_emis_green,is) /=                         &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%emissivity(m,ind_pav_green) =                                         &
                              building_surface_pars_f%pars(ind_s_emis_green,is)

                 IF ( building_surface_pars_f%pars(ind_s_emis_win,is) /=                           &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%emissivity(m,ind_wat_win) =                                           &
                              building_surface_pars_f%pars(ind_s_emis_win,is)

                 IF ( building_surface_pars_f%pars(ind_s_trans,is) /=                              &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%transmissivity(m) =                                                   &
                              building_surface_pars_f%pars(ind_s_trans,is)

                 IF ( building_surface_pars_f%pars(ind_s_z0,is) /=                                 &
                      building_surface_pars_f%fill )                                               &
                    surf_usm%z0(m) = building_surface_pars_f%pars(ind_s_z0,is)

                 IF ( building_surface_pars_f%pars(ind_s_z0qh,is) /=                               &
                      building_surface_pars_f%fill )                                               &
                 THEN
                    surf_usm%z0q(m) = building_surface_pars_f%pars(ind_s_z0qh,is)
                    surf_usm%z0h(m) = building_surface_pars_f%pars(ind_s_z0qh,is)
                 ENDIF

                 EXIT ! Surface was found and processed
              ENDIF
           ENDDO
        ENDDO
     ENDIF
!
!--  Initialize albedo type via given type from static input file. Please note, even though
!--  the albedo type has been already given by the pars, albedo_type overwrites these values.
     IF ( albedo_type_f%from_file )  THEN
        DO  m = 1, surf_usm%ns
           i = surf_usm%i(m)
           j = surf_usm%j(m)
           IF ( albedo_type_f%var(j,i) /= albedo_type_f%fill )                                     &
              surf_usm%albedo_type(m,:) = albedo_type_f%var(j,i)
        ENDDO
     ENDIF
!
!--  Check if material fractions of surfaces sum up to 1.
     relative_fraction_error = .FALSE.
     DO  m = 1, surf_usm%ns
        IF ( ( ABS( SUM( surf_usm%frac(m,:) ) - 1.0_wp ) ) > 1.0E-5_wp )  THEN
           relative_fraction_error = .TRUE.
        ENDIF
     ENDDO

#if defined( __parallel )
     CALL MPI_ALLREDUCE( MPI_IN_PLACE, relative_fraction_error, 1, MPI_LOGICAL, MPI_LOR, comm2d,   &
                         ierr )
#endif

     IF ( relative_fraction_error )  THEN
        message_string = 'relative material fractions do not sum-up to one at some surfaces'
        CALL message( 'usm_init', 'USM0011', 2, 2, 0, 6, 0 )
     ENDIF

!
!--  Initialization of the wall/roof materials.
     CALL usm_init_wall_heat_model()

!
!--  Init skin layer properties (can be done after initialization of wall layers).
     DO  m = 1, surf_usm%ns
         surf_usm%c_surface(m)           = surf_usm%rho_c_wall(nzb_wall,m)      *                  &
                                           surf_usm%dz_wall(nzb_wall,m)         * 0.25_wp
         surf_usm%lambda_surf(m)         = surf_usm%lambda_h(nzb_wall,m)        *                  &
                                           surf_usm%ddz_wall(nzb_wall,m)        * 2.0_wp
         surf_usm%c_surface_green(m)     = surf_usm%rho_c_wall(nzb_wall,m)      *                  &
                                           surf_usm%dz_wall(nzb_wall,m)         * 0.25_wp
         surf_usm%lambda_surf_green(m)   = surf_usm%lambda_h_green(nzb_wall,m)  *                  &
                                           surf_usm%ddz_green(nzb_wall,m)       * 2.0_wp
         surf_usm%c_surface_window(m)    = surf_usm%rho_c_window(nzb_wall,m)    *                  &
                                           surf_usm%dz_window(nzb_wall,m)       * 0.25_wp
         surf_usm%lambda_surf_window(m)  = surf_usm%lambda_h_window(nzb_wall,m) *                  &
                                           surf_usm%ddz_window(nzb_wall,m)      * 2.0_wp
     ENDDO

!
!-- Check for consistent initialization.
!-- Check if roughness length for momentum, heat, or moisture exceed surface-layer height and
!-- limit local roughness length where necessary (if allowed). If limited, give an informative
!-- message only once in order to avoid the job protocol to be messed-up with messages.
    flag_exceed_z0  = .FALSE.
    flag_exceed_z0h = .FALSE.
    flag_exceed_z0q = .FALSE.
    DO  m = 1, surf_usm%ns

       IF ( surf_usm%z0(m) >= 0.5 * surf_usm%z_mo(m) )  THEN
          flag_exceed_z0 = .TRUE.
          IF ( allow_roughness_limitation )  THEN
             surf_usm%z0(m) = 0.5_wp * surf_usm%z_mo(m)
          ELSE
             exit_index = m
             EXIT
          ENDIF
       ENDIF

       IF ( surf_usm%z0h(m) >= surf_usm%z_mo(m) )  THEN
          flag_exceed_z0h = .TRUE.
          IF ( allow_roughness_limitation )  THEN
             surf_usm%z0h(m) = 0.5_wp * surf_usm%z_mo(m)
          ELSE
             exit_index = m
             EXIT
          ENDIF
       ENDIF

       IF ( surf_usm%z0q(m) >= surf_usm%z_mo(m) )  THEN
          flag_exceed_z0q = .TRUE.
          IF ( allow_roughness_limitation )  THEN
             surf_usm%z0q(m) = 0.5_wp * surf_usm%z_mo(m)
          ELSE
             exit_index = m
             EXIT
          ENDIF
       ENDIF

    ENDDO

    IF ( flag_exceed_z0  .AND.  .NOT. allow_roughness_limitation )  THEN
       WRITE( message_string, '(A,I6,A,I6,A)' )                                                    &
              'z0 exceeds 0.5 * surface-layer height at building surface grid point (i,j) = (',    &
               surf_usm%i(exit_index)+surf_usm%ioff(exit_index), ',',                              &
               surf_usm%j(exit_index)+surf_usm%joff(exit_index), ')'
       CALL message( 'usm_init', 'USM0012', 2, 2, myid, 6, 0 )
    ENDIF
#if defined( __parallel )
    CALL MPI_ALLREDUCE( MPI_IN_PLACE, flag_exceed_z0, 1, MPI_LOGICAL, MPI_LOR, comm2d, ierr)
#endif
    IF ( flag_exceed_z0 )  THEN
       WRITE( message_string, * ) 'z0 exceeds 0.5 * surface-layer height at building surface(s)' //&
                                  ' and is limited to that height'
       CALL message( 'usm_init', 'USM0013', 0, 0, 0, 6, 0 )
    ENDIF

    IF ( flag_exceed_z0h  .AND.  .NOT. allow_roughness_limitation )  THEN
       WRITE( message_string, '(A,I6,A,I6,A)' )                                                    &
              'z0h exceeds 0.5 * surface-layer height at building surface grid point (i,j) = (',   &
               surf_usm%i(exit_index)+surf_usm%ioff(exit_index), ',',                              &
               surf_usm%j(exit_index)+surf_usm%joff(exit_index), ')'
       CALL message( 'usm_init', 'USM0012', 2, 2, myid, 6, 0 )
    ENDIF
#if defined( __parallel )
    CALL MPI_ALLREDUCE( MPI_IN_PLACE, flag_exceed_z0h, 1, MPI_LOGICAL, MPI_LOR, comm2d, ierr)
#endif
    IF ( flag_exceed_z0h )  THEN
       WRITE( message_string, * ) 'z0h exceeds 0.5 * surface-layer height at building surface(s)'//&
                                  ' and is limited to that height'
       CALL message( 'usm_init', 'USM0013', 0, 0, 0, 6, 0 )
    ENDIF

    IF ( flag_exceed_z0q  .AND.  .NOT. allow_roughness_limitation )  THEN
       WRITE( message_string, '(A,I6,A,I6,A)' )                                                    &
              'z0q exceeds 0.5 * surface-layer height at building surface grid point (i,j) = (',   &
               surf_usm%i(exit_index)+surf_usm%ioff(exit_index), ',',                              &
               surf_usm%j(exit_index)+surf_usm%joff(exit_index), ')'
       CALL message( 'usm_init', 'USM0012', 2, 2, myid, 6, 0 )
    ENDIF
#if defined( __parallel )
    CALL MPI_ALLREDUCE( MPI_IN_PLACE, flag_exceed_z0q, 1, MPI_LOGICAL, MPI_LOR, comm2d, ierr)
#endif
    IF ( flag_exceed_z0q )  THEN
       WRITE( message_string, * ) 'z0q exceeds 0.5 * surface-layer height at building surface(s)'//&
                                  ' and is limited to that height'
       CALL message( 'usm_init', 'USM0013', 0, 0, 0, 6, 0 )
    ENDIF

!
!--  Intitialization of the surface and wall/ground/roof temperature. These actions must not be done
!--  in restart runs or when spinup data is read.
     IF ( TRIM( initializing_actions ) /= 'read_restart_data'  .AND.                               &
          .NOT.  read_spinup_data )  THEN
        DO  m = 1, surf_usm%ns
           i = surf_usm%i(m)
           j = surf_usm%j(m)
           k = surf_usm%k(m)

           t_surf_wall%val(m)     = pt(k,j,i) * exner(k)
           t_surf_window%val(m)   = pt(k,j,i) * exner(k)
           t_surf_green%val(m)    = pt(k,j,i) * exner(k)
           surf_usm%pt_surface(m) = pt(k,j,i) * exner(k)
        ENDDO
!
!--      For the sake of correct initialization, set also q_surface.
!--      Note, at urban surfaces q_surface is initialized with 0.
         IF ( humidity )  THEN
            DO  m = 1, surf_usm%ns
               surf_usm%q_surface(m) = 0.0_wp
            ENDDO
         ENDIF
!
!--      Initial values for t_wall.
!--      Outer value is set to surface temperature, inner value is set to wall_inner_temperature
!--      and profile is logaritmic (linear in nz).
!--      Again, initialization is separated between roofs and walls. Start with roofs.
         DO  m = 1, surf_usm%ns
            IF ( surf_usm%upward(m)  .OR.  surf_usm%downward(m) )  THEN
               tin = roof_inner_temperature
            ELSE
               tin = wall_inner_temperature
            ENDIF
            twin = window_inner_temperature

            DO k = nzb_wall, nzt_wall+1
                c = REAL( k - nzb_wall, wp ) / REAL( nzt_wall + 1 - nzb_wall, wp )

                t_wall%val(k,m)   = ( 1.0_wp - c ) * t_surf_wall%val(m) + c * tin
                t_window%val(k,m) = ( 1.0_wp - c ) * t_surf_window%val(m) + c * twin
                t_green%val(k,m)  = t_surf_wall%val(m)
            ENDDO
         ENDDO
     ENDIF

!
!--  Possibly DO user-defined actions (e.g. define heterogeneous wall surface)
     CALL user_init_urban_surface

!
!--  Initialize prognostic values for the first timestep
     t_surf_wall_p   = t_surf_wall
     t_surf_window_p = t_surf_window
     t_surf_green_p  = t_surf_green

     swc_p      = swc
     t_wall_p   = t_wall
     t_window_p = t_window
     t_green_p  = t_green

!
!-- Set initial values for prognostic soil quantities
    IF ( TRIM( initializing_actions ) /= 'read_restart_data'  .AND.  .NOT. read_spinup_data )  THEN
       m_liq_usm%val = 0.0_wp
    ENDIF
    m_liq_usm_p%val = m_liq_usm%val
!
!-- Set initial values for diagnostic quantities
    surf_usm%c_liq    = 0.0_wp
    surf_usm%qsws_liq = 0.0_wp
    surf_usm%qsws_veg = 0.0_wp
!
!-- Finally, set maximum allowed timestep to a huge number, so that it does not affect the
!-- timestep calculation at the very first call.
    surf_usm%dt_max = HUGE( 1.0_wp )

    CALL cpu_log( log_point_s(78), 'usm_init', 'stop' )

    IF ( debug_output )  CALL debug_message( 'usm_init', 'end' )

 END SUBROUTINE usm_init


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!
!> Wall model as part of the urban surface model. The model predicts vertical and horizontal
!> wall / roof temperatures and window layer temperatures. No window layer temperature calculactions
!> during spinup to increase possible timestep.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE usm_wall_heat_model( during_spinup )


    IMPLICIT NONE

    LOGICAL, INTENT(IN) ::  during_spinup  !< if true, no calculation of window temperatures
    LOGICAL             ::  runge_l        !< dummy flag to indicate RK timestepping scheme

    INTEGER(iwp) ::  kw !< grid index - wall depth
    INTEGER(iwp) ::  m  !< running index for surface elements

    REAL(wp) ::  win_absorp        !< absorption coefficient from transmissivity
    REAL(wp) ::  win_nonrefl_1side !< non-reflected fraction after outer glass boundary

    REAL(wp), DIMENSION(nzb_wall:nzt_wall) ::  max_dt_green !< maximum allowed timestep at green surfaces according diffusion criterion
    REAL(wp), DIMENSION(nzb_wall:nzt_wall) ::  max_dt_wall  !< maximum allowed timestep at walls according diffusion criterion
    REAL(wp), DIMENSION(nzb_wall:nzt_wall) ::  max_dt_win   !< maximum allowed timestep at windows according diffusion criterion
    REAL(wp), DIMENSION(nzb_wall:nzt_wall) ::  wall_mod     !< relaxation factor to reduce wall conductivity during spinup phase

    REAL(wp), DIMENSION(1:surf_usm%ns) ::  fac_veg           !< pre-calcualated factor for vegetation fraction
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  fac_wall          !< pre-calcualated factor for wall fraction

    REAL(wp), DIMENSION(nzb_wall:nzt_wall,1:surf_usm%ns) ::  wtend   !< computed tendency for wall fraction
    REAL(wp), DIMENSION(nzb_wall:nzt_wall,1:surf_usm%ns) ::  wintend !< computed tendency for window fraction

    TYPE(surf_type), POINTER ::  surf !< surface-date type variable


    runge_l = (timestep_scheme(1:5) == 'runge')

    IF ( debug_output_timestep )  THEN
       WRITE( debug_string, * ) 'usm_wall_heat_model: '
       CALL debug_message( debug_string, 'start' )
    ENDIF
!
!-- Determine relaxation factor used during soil-wall spinup.
    wall_mod = 1.0_wp
    IF ( usm_wall_mod  .AND.  during_spinup )  THEN
       DO  kw=nzb_wall, nzb_wall+1
          wall_mod(kw) = 0.1_wp
       ENDDO
    ENDIF

    surf => surf_usm
!
!-- Prognostic equation for ground/roof temperature t_wall
    wtend = 0.0_wp

!
!-- Set minimum allowed timestep array for the tile fractions to a huge number.
    max_dt_green = HUGE( 1.0_wp )
    max_dt_wall  = HUGE( 1.0_wp )
    max_dt_win   = HUGE( 1.0_wp )
!
!-- Compute fractions.
    fac_wall = 0.0_wp
    fac_veg  = 0.0_wp
    !$OMP PARALLEL DO PRIVATE (m) SCHEDULE (STATIC)
    DO  m = 1, surf%ns
       IF ( surf%frac(m,ind_veg_wall) + surf%frac(m,ind_pav_green) > 0.0_wp )  THEN
          fac_wall(m) = surf%frac(m,ind_veg_wall) /                                                &
                        ( surf%frac(m,ind_veg_wall) + surf%frac(m,ind_pav_green) )
          fac_veg(m) = surf%frac(m,ind_pav_green) /                                                &
                       ( surf%frac(m,ind_veg_wall) + surf%frac(m,ind_pav_green) )
       ENDIF
    ENDDO
!
!-- If indoor model is used inner wall layer is calculated by using iwghf (indoor
!-- wall ground heat flux)
    IF ( .NOT. indoor_model ) THEN
       !$OMP PARALLEL DO PRIVATE (m) SCHEDULE (STATIC)
       DO  m = 1, surf%ns
          surf%iwghf_eb(m) = surf%lambda_h(nzt_wall,m)  * wall_mod(nzt_wall) *                     &
                             ( t_wall%val(nzt_wall+1,m) - t_wall%val(nzt_wall,m) ) *               &
                             surf%ddz_wall_center(nzt_wall,m)

          surf%iwghf_eb_window(m) = surf%lambda_h_window(nzt_wall,m) *                             &
                                    ( t_window%val(nzt_wall+1,m) - t_window%val(nzt_wall,m) ) *    &
                                    surf%ddz_window_center(nzt_wall,m)
       ENDDO
    ENDIF
!
!-- Compute tendency terms for wall fraction.
    !$OMP PARALLEL DO PRIVATE (kw, m, wtend) SCHEDULE (STATIC)
    DO  m = 1, surf%ns
       wtend(nzb_wall,m) = ( 1.0_wp / surf%rho_c_wall(nzb_wall,m) ) *                              &
                           ( surf%lambda_h_layer(nzb_wall,m) * wall_mod(nzb_wall) *                &
                           ( t_wall%val(nzb_wall+1,m) - t_wall%val(nzb_wall,m) ) *                 &
                             surf%ddz_wall_center(nzb_wall,m)                                      &
                         + fac_wall(m) * surf%wghf_eb(m)                                           &
                         - fac_veg(m) * ( surf%lambda_h_green(nzt_wall,m) * wall_mod(nzt_wall) *   &
                                          surf%ddz_green_center(nzt_wall,m)                        &
                                        + surf%lambda_h_layer(nzb_wall,m) * wall_mod(nzb_wall) *   &
                                          surf%ddz_wall_center(nzb_wall,m)                         &
                                        ) /                                                        &
                                        ( surf%dz_green_center(nzt_wall,m) +                       &
                                          surf%dz_wall_center(nzb_wall,m) ) * 4.0_wp *             &
                                        ( t_wall%val(nzb_wall,m) - t_green%val(nzt_wall,m) )       &
                           ) * surf%ddz_wall(nzb_wall,m)



       DO  kw = nzb_wall+1, nzt_wall-1
          wtend(kw,m) = ( 1.0_wp / surf%rho_c_wall(kw,m) ) *                                       &
                        ( surf%lambda_h_layer(kw,m) * wall_mod(kw) *                               &
                        ( t_wall%val(kw+1,m) - t_wall%val(kw,m) ) *                                &
                        surf%ddz_wall_center(kw,m) -                                               &
                        surf%lambda_h_layer(kw-1,m) * wall_mod(kw-1) *                             &
                        ( t_wall%val(kw,m) - t_wall%val(kw-1,m) ) *                                &
                        surf%ddz_wall_center(kw-1,m)                                               &
                        ) * surf%ddz_wall(kw,m)
       ENDDO
       wtend(nzt_wall,m) = ( 1.0_wp / surf%rho_c_wall(nzt_wall,m) ) *                              &
                           ( -surf%lambda_h_layer(nzt_wall-1,m) * wall_mod(nzt_wall-1) *           &
                             ( t_wall%val(nzt_wall,m) - t_wall%val(nzt_wall-1,m) ) *               &
                               surf%ddz_wall_center(nzt_wall-1,m) + surf%iwghf_eb(m)               &
                           ) * surf%ddz_wall(nzt_wall,m)

       DO  kw = nzb_wall, nzt_wall
          t_wall_p%val(kw,m) = t_wall%val(kw,m) + dt_3d * ( tsc(2) * wtend(kw,m) +                 &
                                                            tsc(3) * surf%tt_wall_m(kw,m) )
       ENDDO
    ENDDO
!
!-- Compute tendency terms for all window fractions.
!-- Skip this during spinup. During the spinup, the tempeature inside window layers is not
!-- calculated to make larger timesteps possible.
    IF ( .NOT. during_spinup )  THEN
       wintend = 0.0_wp

       !$OMP PARALLEL DO PRIVATE (kw, m, wintend, win_absorp) SCHEDULE (STATIC)
       DO  m = 1, surf%ns
!
!--       Reflectivity in glass windows is considered as equal on frontal and rear side of the
!--       glass, which together make total reflectivity (albedo for win fraction).
          win_nonrefl_1side = 1.0_wp - ( surf%albedo(m,ind_wat_win) + surf%transmissivity(m)       &
                                      + 1.0_wp                                                     &
                                      - SQRT( ( surf%albedo(m,ind_wat_win)                         &
                                                + surf%transmissivity(m) + 1.0_wp )**2             &
                                              - 4.0_wp * surf%albedo(m,ind_wat_win) ) ) / 2.0_wp
!
!--       Absorption coefficient is calculated using zw from internal tranmissivity, which only
!--       considers absorption without the effects of reflection.
          win_absorp = -LOG( ( surf%transmissivity(m) + surf%albedo(m,ind_wat_win) - 1.0_wp        &
                               + win_nonrefl_1side ) / win_nonrefl_1side )                         &
                       / surf%zw_window(nzt_wall,m)
!
!--       Prognostic equation for ground/roof window temperature t_window takes absorption
!--       of shortwave radiation into account
          wintend(nzb_wall,m) = ( 1.0_wp / surf%rho_c_window(nzb_wall,m) )                         &
                                * ( surf%lambda_h_window_layer(nzb_wall,m)                         &
                                * ( t_window%val(nzb_wall+1,m) - t_window%val(nzb_wall,m) )        &
                                * surf%ddz_window_center(nzb_wall,m)                               &
                                + surf%wghf_eb_window(m)                                           &
                                + surf%rad_sw_in(m) * win_nonrefl_1side                            &
                                * ( 1.0_wp - EXP( -win_absorp * surf%zw_window(nzb_wall,m) ) )     &
                                  ) * surf%ddz_window(nzb_wall,m)


          DO  kw = nzb_wall+1, nzt_wall-1
             wintend(kw,m) = ( 1.0_wp / surf%rho_c_window(kw,m) )                                  &
                           * ( surf%lambda_h_window_layer(kw,m)                                    &
                           * ( t_window%val(kw+1,m) - t_window%val(kw,m) )                         &
                           * surf%ddz_window_center(kw,m)                                          &
                           - surf%lambda_h_window_layer(kw-1,m)                                    &
                           * ( t_window%val(kw,m) - t_window%val(kw-1,m) )                         &
                           * surf%ddz_window_center(kw-1,m)                                        &
                           + surf%rad_sw_in(m) * win_nonrefl_1side                                 &
                           * ( EXP( -win_absorp * surf%zw_window(kw-1,m) )                         &
                             - EXP( -win_absorp * surf%zw_window(kw,m) ) )                         &
                             ) * surf%ddz_window(kw,m)

          ENDDO

          wintend(nzt_wall,m) = ( 1.0_wp / surf%rho_c_window(nzt_wall,m) )                         &
                                 * ( -surf%lambda_h_window_layer(nzt_wall-1,m)                     &
                                 * ( t_window%val(nzt_wall,m) - t_window%val(nzt_wall-1,m) )       &
                                 * surf%ddz_window_center(nzt_wall-1,m)                            &
                                 + surf%iwghf_eb_window(m)                                         &
                                 + surf%rad_sw_in(m) * win_nonrefl_1side                           &
                                 * ( EXP( -win_absorp * surf%zw_window(nzt_wall-1,m) )             &
                                   - EXP( -win_absorp * surf%zw_window(nzt_wall,m) ) )             &
                                   ) * surf%ddz_window(nzt_wall,m)

          DO  kw = nzb_wall, nzt_wall
             t_window_p%val(kw,m) = t_window%val(kw,m) + dt_3d * ( tsc(2) * wintend(kw,m) +        &
                                                                   tsc(3) * surf%tt_window_m(kw,m) )
          ENDDO
       ENDDO
    ENDIF
!
!-- Calculate weighted Runge-Kutta t_wall tendencies for the next substep.
    IF ( runge_l )  THEN
       IF ( intermediate_timestep_count == 1 )  THEN
          !$OMP PARALLEL DO PRIVATE (kw, m) SCHEDULE (STATIC)
          DO  m = 1, surf%ns
             DO  kw = nzb_wall, nzt_wall
                surf%tt_wall_m(kw,m) = wtend(kw,m)
             ENDDO
          ENDDO
       ELSEIF ( intermediate_timestep_count < intermediate_timestep_count_max )  THEN
          !$OMP PARALLEL DO PRIVATE (kw, m) SCHEDULE (STATIC)
          DO  m = 1, surf%ns
             DO  kw = nzb_wall, nzt_wall
                surf%tt_wall_m(kw,m) = -9.5625_wp * wtend(kw,m) +                                  &
                                        5.3125_wp * surf%tt_wall_m(kw,m)
             ENDDO
          ENDDO
       ENDIF
    ENDIF
!
!-- Calculate weighted Runge-Kutta t_window tendencies for the next substep. Skip this during
!-- the spinup phase.
    IF ( .NOT. during_spinup )  THEN
       IF ( runge_l )  THEN
          IF ( intermediate_timestep_count == 1 )  THEN
             !$OMP PARALLEL DO PRIVATE (kw, m) SCHEDULE (STATIC)
             DO  m = 1, surf%ns
                DO  kw = nzb_wall, nzt_wall
                   surf%tt_window_m(kw,m) = wintend(kw,m)
                ENDDO
             ENDDO
          ELSEIF ( intermediate_timestep_count < intermediate_timestep_count_max )  THEN
             !$OMP PARALLEL DO PRIVATE (kw, m) SCHEDULE (STATIC)
             DO  m = 1, surf%ns
                DO  kw = nzb_wall, nzt_wall
                   surf%tt_window_m(kw,m) = -9.5625_wp * wintend(kw,m) +                           &
                                             5.3125_wp * surf%tt_window_m(kw,m)
                ENDDO
             ENDDO
          ENDIF
       ENDIF
    ENDIF
!
!-- Finally, evaluate maximum allowed timestep. This will be later used to limit the model
!-- timestep accordingly.
    IF ( intermediate_timestep_count == intermediate_timestep_count_max )  THEN
       !$OMP PARALLEL DO PRIVATE (kw, m) SCHEDULE (STATIC)
       DO  m = 1, surf%ns
          DO  kw = nzb_wall, nzt_wall
             IF ( surf%frac(m,ind_veg_wall) > 0.0_wp )    THEN
                max_dt_wall(kw) = surf%rho_c_wall(kw,m) / surf%lambda_h_layer(kw,m) *              &
                                  wall_mod(kw) * ( surf%dz_wall(kw,m) )**2
             ELSE
                max_dt_wall(kw) = HUGE( 1.0_wp )
             ENDIF
             IF ( surf%frac(m,ind_wat_win) > 0.0_wp )    THEN
                max_dt_win(kw)  = surf%rho_c_window(kw,m) / surf%lambda_h_window_layer(kw,m) *     &
                                  ( surf%dz_window(kw,m) )**2
             ELSE
                max_dt_win(kw) = HUGE( 1.0_wp )
             ENDIF
             IF ( surf%frac(m,ind_pav_green) > 0.0_wp )     THEN
                max_dt_green(kw) = surf%rho_c_total_green(kw,m) / surf%lambda_h_green(kw,m) *      &
                                   ( surf%dz_green(kw,m) )**2
             ELSE
                max_dt_green(kw) = HUGE( 1.0_wp )
             ENDIF
          ENDDO
          surf%dt_max(m) = MIN( MINVAL( max_dt_wall(:) ), MINVAL( max_dt_win(:) ),                 &
                                MINVAL( max_dt_green(:) ) )
       ENDDO
    ENDIF

    IF ( debug_output_timestep )  THEN
       WRITE( debug_string, * ) 'usm_wall_heat_model: ', during_spinup
       CALL debug_message( debug_string, 'end' )
    ENDIF

 END SUBROUTINE usm_wall_heat_model

!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!
!> Green and substrate model as part of the urban surface model. The model predicts ground
!> temperatures.
!>
!> Important: green-heat model crashes due to unknown reason. Green fraction is thus set to zero
!> (in favor of wall fraction).
!> Note, usm_green_heat_model has not been vectorized yet.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE usm_green_heat_model

    IMPLICIT NONE

    INTEGER(iwp)  ::  i, j, k, kw, m      !< running indices

    LOGICAL  ::  conserve_water_content = .TRUE.  !<

    REAL(wp)  ::  drho_l_lv               !< frequently used parameter
    REAL(wp)  ::  h_vg                    !< Van Genuchten coef. h
    REAL(wp)  ::  ke, lambda_h_green_sat  !< heat conductivity for saturated soil

    REAL(wp), DIMENSION(nzb_wall:nzt_wall)   ::  gtend,tend         !< tendency
    REAL(wp), DIMENSION(nzb_wall:nzt_wall)   ::  root_extr_green    !<

    REAL(wp), DIMENSION(nzb_wall:nzt_wall+1) ::  gamma_green_temp   !< temp. gamma
    REAL(wp), DIMENSION(nzb_wall:nzt_wall+1) ::  lambda_green_temp  !< temp. lambda

    TYPE(surf_type), POINTER                 ::  surf               !< surface-date type variable

    IF ( debug_output_timestep )  THEN
       WRITE( debug_string, * ) 'usm_green_heat_model: '
       CALL debug_message( debug_string, 'start' )
    ENDIF

    drho_l_lv = 1.0_wp / (rho_l * l_v)
!
!-- Set pointer to urban-surface structure. Note, with all surfaces in one array this is
!-- actually not necessary any more. However, for future developments when land- and urban-
!-- surface model will be merged, this will become necessary again.
    surf => surf_usm

!-- Set tendency array for soil moisture to zero
    IF ( surf%ns > 0 )  THEN
       IF ( intermediate_timestep_count == 1 )  surf%tswc_m = 0.0_wp
    ENDIF

    !$OMP PARALLEL DO PRIVATE (m, i, j, k, kw, lambda_h_green_sat, ke, lambda_green_temp,      &
    !$OMP&  gtend, tend, h_vg, gamma_green_temp, m_total, root_extr_green) SCHEDULE (STATIC)
    DO  m = 1, surf%ns
!
!--    For green fraction at upward-facing walls
       IF ( surf%frac(m,ind_pav_green) > 0.0_wp  .AND.  surf%upward(m) )  THEN
!
!--      Obtain indices
          i = surf%i(m)
          j = surf%j(m)
          k = surf%k(m)

          DO  kw = nzb_wall, nzt_wall
!
!--          Calculate volumetric heat capacity of the soil, taking into account water content
             surf%rho_c_total_green(kw,m) = surf%rho_c_green(kw,m) * ( 1.0_wp - swc_sat%val(kw,m) )&
                                            + rho_c_water * swc%val(kw,m)
!
!--          Calculate soil heat conductivity at the center of the soil layers
             lambda_h_green_sat = lambda_h_green_sm**( 1.0_wp - swc_sat%val(kw,m) )                &
                                  * lambda_h_water**swc%val(kw,m)

             ke = 1.0_wp + LOG10( MAX( 0.1_wp,swc%val(kw,m) / swc_sat%val(kw,m) ) )

             lambda_green_temp(kw) = ke * (lambda_h_green_sat - lambda_h_green_dry)                &
                                     + lambda_h_green_dry

          ENDDO
          lambda_green_temp(nzt_wall+1) = lambda_green_temp(nzt_wall)


!
!--       Calculate soil heat conductivity (lambda_h) at the _center level using weighting
          DO  kw = nzb_wall, nzt_wall-1
             surf%lambda_h_green(kw,m) = ( lambda_green_temp(kw)  * surf%dz_green(kw,m)            &
                                         + lambda_green_temp(kw+1) * surf%dz_green(kw+1,m)         &
                                         ) * 0.5_wp * surf%ddz_green_center(kw,m)
          ENDDO
          surf%lambda_h_green(nzt_wall,m) = lambda_green_temp(nzt_wall)

          t_green%val(nzt_wall+1,m) = t_wall%val(nzb_wall,m)
!
!--       Prognostic equation for ground/roof temperature t_green
          gtend(:) = 0.0_wp
          gtend(nzb_wall) = ( 1.0_wp / surf%rho_c_total_green(nzb_wall,m) )                        &
                            * ( surf%lambda_h_green(nzb_wall,m)                                    &
                                * ( t_green%val(nzb_wall+1,m) - t_green%val(nzb_wall,m) )          &
                                * surf%ddz_green_center(nzb_wall,m)                                &
                                + surf%wghf_eb_green(m)                                            &
                              ) * surf%ddz_green(nzb_wall,m)

          DO  kw = nzb_wall+1, nzt_wall
             gtend(kw) = ( 1.0_wp / surf%rho_c_total_green(kw,m) )                                 &
                         * ( surf%lambda_h_green(kw,m)                                             &
                             * ( t_green%val(kw+1,m) - t_green%val(kw,m) )                         &
                             * surf%ddz_green_center(kw,m)                                         &
                             - surf%lambda_h_green(kw-1,m)                                         &
                             * ( t_green%val(kw,m) - t_green%val(kw-1,m) )                         &
                             * surf%ddz_green_center(kw-1,m)                                       &
                           ) * surf%ddz_green(kw,m)
          ENDDO

          t_green_p%val(nzb_wall:nzt_wall,m) = t_green%val(nzb_wall:nzt_wall,m)                    &
                         + dt_3d * ( tsc(2) * gtend(nzb_wall:nzt_wall) + tsc(3)                    &
                         * surf%tt_green_m(nzb_wall:nzt_wall,m) )


!
!--       Calculate t_green tendencies for the next Runge-Kutta step
          IF ( timestep_scheme(1:5) == 'runge' )  THEN
              IF ( intermediate_timestep_count == 1 )  THEN
                 DO  kw = nzb_wall, nzt_wall
                    surf%tt_green_m(kw,m) = gtend(kw)
                 ENDDO
              ELSEIF ( intermediate_timestep_count < intermediate_timestep_count_max )  THEN
                  DO  kw = nzb_wall, nzt_wall
                     surf%tt_green_m(kw,m) = -9.5625_wp * gtend(kw) +                              &
                                              5.3125_wp * surf%tt_green_m(kw,m)
                  ENDDO
              ENDIF
          ENDIF

          DO  kw = nzb_wall, nzt_wall

!
!--          Calculate soil diffusivity at the center of the soil layers
             lambda_green_temp(kw) = ( - b_ch * surf%gamma_w_green_sat(kw,m) * psi_sat             &
                                       / swc_sat%val(kw,m) )                                       &
                                     * ( MAX( swc%val(kw,m), wilt%val(kw,m) )                      &
                                       / swc_sat%val(kw,m) )**( b_ch + 2.0_wp )

!
!--          Parametrization of Van Genuchten
             IF ( soil_type /= 7 )  THEN
!
!--             Calculate the hydraulic conductivity after Van Genuchten (1980)
                h_vg = ( ( (swc_res%val(kw,m) - swc_sat%val(kw,m))                                 &
                           / ( swc_res%val(kw,m) -                                                 &
                           MAX( swc%val(kw,m), wilt%val(kw,m) ) ) )**                              &
                           ( surf%n_vg_green(m) / (surf%n_vg_green(m) - 1.0_wp ) )                 &
                           - 1.0_wp                                                                &
                       )** ( 1.0_wp / surf%n_vg_green(m) ) / surf%alpha_vg_green(m)


                gamma_green_temp(kw) = surf%gamma_w_green_sat(kw,m)                                &
                                       * ( ( ( 1.0_wp + ( surf%alpha_vg_green(m) * h_vg )**        &
                                             surf%n_vg_green(m) )**                                &
                                             ( 1.0_wp - 1.0_wp / surf%n_vg_green(m) )              &
                                             - ( surf%alpha_vg_green(m) * h_vg )**                 &
                                             ( surf%n_vg_green(m) - 1.0_wp) )**2                   &
                                         ) / ( ( 1.0_wp + ( surf%alpha_vg_green(m) * h_vg )**      &
                                               surf%n_vg_green(m) )**                              &
                                               ( ( 1.0_wp  - 1.0_wp / surf%n_vg_green(m) )         &
                                                 *( surf%l_vg_green(m) + 2.0_wp) )                 &
                                             )

!
!--          Parametrization of Clapp & Hornberger
             ELSE
                gamma_green_temp(kw) = surf%gamma_w_green_sat(kw,m) * ( swc%val(kw,m)              &
                                       / swc_sat%val(kw,m) )**( 2.0_wp * b_ch + 3.0_wp )
             ENDIF

          ENDDO

!
!--       Prognostic equation for soil moisture content. Only performed, when humidity is enabled in
!--       the atmosphere
          IF ( humidity )  THEN
!
!--          Calculate soil diffusivity (lambda_w) at the _center level using weighting
!--          To do: replace this with ECMWF-IFS Eq. 8.81
             DO  kw = nzb_wall, nzt_wall-1

                surf%lambda_w_green(kw,m) = ( lambda_green_temp(kw)  * surf%dz_green(kw,m)         &
                                            + lambda_green_temp(kw+1) * surf%dz_green(kw+1,m)      &
                                            ) * 0.5_wp * surf%ddz_green_center(kw,m)
                surf%gamma_w_green(kw,m)  = ( gamma_green_temp(kw)  * surf%dz_green(kw,m)          &
                                            + gamma_green_temp(kw+1) * surf%dz_green(kw+1,m)       &
                                            ) * 0.5_wp * surf%ddz_green_center(kw,m)

             ENDDO

!
!--          In case of a closed bottom (= water content is conserved), set hydraulic conductivity
!--          to zero so that no water will be lost in the bottom layer.
             IF ( conserve_water_content )  THEN
                surf%gamma_w_green(kw,m) = 0.0_wp
             ELSE
                surf%gamma_w_green(kw,m) = gamma_green_temp(nzt_wall)
             ENDIF

!--          The root extraction (= root_extr * qsws_veg / (rho_l * l_v)) ensures the mass
!--          conservation for water. The transpiration of plants equals the cumulative withdrawals
!--          by the roots in the soil. The scheme takes into account the availability of water in
!--          the soil layers as well as the root fraction in the respective layer. Layer with
!--          moisture below wilting point will not contribute, which reflects the preference of
!--          plants to take water from moister layers.

!
!--          Calculate the root extraction (ECMWF 7.69, the sum of root_extr = 1). The energy
!--          balance solver guarantees a positive transpiration, so that there is no need for an
!--          additional check.
             m_total = 0.0_wp
             DO  kw = nzb_wall, nzt_wall
                 IF ( swc%val(kw,m) > wilt%val(kw,m) )  THEN
                    m_total = m_total + rootfr%val(kw,m) * swc%val(kw,m)
                 ENDIF
             ENDDO

             IF ( m_total > 0.0_wp )  THEN
                DO  kw = nzb_wall, nzt_wall
                   IF ( swc%val(kw,m) > wilt%val(kw,m) )  THEN
                      root_extr_green(kw) = rootfr%val(kw,m) * swc%val(kw,m) / m_total
                   ELSE
                      root_extr_green(kw) = 0.0_wp
                   ENDIF
                ENDDO
             ENDIF

!
!--          Prognostic equation for soil water content m_soil.
             tend(:) = 0.0_wp

             tend(nzb_wall) = ( surf%lambda_w_green(nzb_wall,m)                                    &
                              * ( swc%val(nzb_wall+1,m) - swc%val(nzb_wall,m) )                    &
                              * surf%ddz_green_center(nzb_wall,m)                                  &
                              - surf%gamma_w_green(nzb_wall,m)                                     &
                              - ( root_extr_green(nzb_wall) * surf%qsws_veg(m)                     & !+ surf%qsws_soil_green(m)
                                ) * drho_l_lv )                                                    &
                              * surf%ddz_green(nzb_wall,m)

             DO  kw = nzb_wall+1, nzt_wall-1
                tend(kw) = ( surf%lambda_w_green(kw,m)                                             &
                             * ( swc%val(kw+1,m) - swc%val(kw,m) )                                 &
                             * surf%ddz_green_center(kw,m)                                         &
                             - surf%gamma_w_green(kw,m)                                            &
                             - surf%lambda_w_green(kw-1,m)                                         &
                             * ( swc%val(kw,m) - swc%val(kw-1,m) )                                 &
                             * surf%ddz_green_center(kw-1,m)                                       &
                             + surf%gamma_w_green(kw-1,m)                                          &
                             - (root_extr_green(kw)                                                &
                             * surf%qsws_veg(m)                                                    &
                             * drho_l_lv)                                                          &
                          ) * surf%ddz_green(kw,m)

             ENDDO
             tend(nzt_wall) = ( - surf%gamma_w_green(nzt_wall,m)                                   &
                                - surf%lambda_w_green(nzt_wall-1,m)                                &
                                * (swc%val(nzt_wall,m) - swc%val(nzt_wall-1,m))                    &
                                * surf%ddz_green_center(nzt_wall-1,m)                              &
                                + surf%gamma_w_green(nzt_wall-1,m)                                 &
                                - ( root_extr_green(nzt_wall)                                      &
                                * surf%qsws_veg(m)                                                 &
                                * drho_l_lv  )                                                     &
                              ) * surf%ddz_green(nzt_wall,m)

             swc_p%val(nzb_wall:nzt_wall,m) = swc%val(nzb_wall:nzt_wall,m) + dt_3d                 &
                                            * ( tsc(2) * tend(:) + tsc(3)                          &
                                                * surf%tswc_m(:,m)                                 &
                                               )

!
!--          Account for dry soils (find a better solution here!)
             DO  kw = nzb_wall, nzt_wall
                IF ( swc_p%val(kw,m) < 0.0_wp )  swc_p%val(kw,m) = 0.0_wp
             ENDDO

!
!--          Calculate m_soil tendencies for the next Runge-Kutta step
             IF ( timestep_scheme(1:5) == 'runge' )  THEN
                IF ( intermediate_timestep_count == 1 )  THEN
                   DO  kw = nzb_wall, nzt_wall
                      surf%tswc_m(kw,m) = tend(kw)
                   ENDDO
                ELSEIF ( intermediate_timestep_count < intermediate_timestep_count_max )  THEN
                   DO  kw = nzb_wall, nzt_wall
                      surf%tswc_m(kw,m) = -9.5625_wp * tend(kw) + 5.3125_wp * surf%tswc_m(kw,m)
                   ENDDO
                ENDIF
             ENDIF
          ENDIF
!
!--    Vertical walls
       ELSEIF ( surf%frac(m,ind_pav_green) > 0.0_wp )  THEN
!
!--            No substrate layer for green walls / only groundbase green walls (ivy i.e.) -> Green layers get
!--            same temperature as first wall layer, therefore no temperature calculations for vertical green
!--            substrate layers now

!
! !
! !--          Obtain indices
!              i = surf%i(m)
!              j = surf%j(m)
!              k = surf%k(m)
!
!              t_green%val(nzt_wall+1,m) = t_wall%val(nzb_wall,m)
! !
! !--          Prognostic equation for green temperature t_green_v
!              gtend(:) = 0.0_wp
!              gtend(nzb_wall) = (1.0_wp / surf%rho_c_green(nzb_wall,m)) *                        &
!                                      ( surf%lambda_h_green(nzb_wall,m) *                        &
!                                        ( t_green%val(nzb_wall+1,m)                              &
!                                        - t_green%val(nzb_wall,m) ) *                            &
!                                        surf%ddz_green(nzb_wall+1,m)                             &
!                                      + surf%wghf_eb(m) ) *                                      &
!                                        surf%ddz_green_stag(nzb_wall,m)
!
!              DO  kw = nzb_wall+1, nzt_wall
!                 gtend(kw) = (1.0_wp / surf%rho_c_green(kw,m))                                   &
!                           * (   surf%lambda_h_green(kw,m)                                       &
!                             * ( t_green%val(kw+1,m) - t_green%val(kw,m) )                       &
!                             * surf%ddz_green(kw+1,m)                                            &
!                           - surf%lambda_h(kw-1,m)                                               &
!                             * ( t_green%val(kw,m) - t_green%val(kw-1,m) )                       &
!                             * surf%ddz_green(kw,m) )                                            &
!                           * surf%ddz_green_stag(kw,m)
!              ENDDO
!
!              t_green_v_p(l)%val(nzb_wall:nzt_wall,m) =                                          &
!                                   t_green%val(nzb_wall:nzt_wall,m)                              &
!                                 + dt_3d * ( tsc(2)                                              &
!                                 * gtend(nzb_wall:nzt_wall) + tsc(3)                             &
!                                 * surf%tt_green_m(nzb_wall:nzt_wall,m) )
!
! !
! !--          Calculate t_green tendencies for the next Runge-Kutta step
!              IF ( timestep_scheme(1:5) == 'runge' )  THEN
!                  IF ( intermediate_timestep_count == 1 )  THEN
!                     DO  kw = nzb_wall, nzt_wall
!                        surf%tt_green_m(kw,m) = gtend(kw)
!                     ENDDO
!                  ELSEIF ( intermediate_timestep_count <                                         &
!                           intermediate_timestep_count_max )  THEN
!                      DO  kw = nzb_wall, nzt_wall
!                         surf%tt_green_m(kw,m) =                                                 &
!                                     - 9.5625_wp * gtend(kw) +                                   &
!                                       5.3125_wp * surf%tt_green_m(kw,m)
!                      ENDDO
!                  ENDIF
!              ENDIF

!
!--       Workaround, set green surface temperature to wall temperature.
          DO  kw = nzb_wall, nzt_wall+1
              t_green%val(kw,m) = t_wall%val(nzb_wall,m)
          ENDDO
!
!--       Workaround, rho_c_total_green is used for calculation of the max_dt_green
          DO  kw = nzb_wall, nzt_wall
              surf%rho_c_total_green(kw,m) = surf%rho_c_green(kw,m)
          ENDDO
       ENDIF
    ENDDO

    IF ( debug_output_timestep )  THEN
       WRITE( debug_string, * ) 'usm_green_heat_model: '
       CALL debug_message( debug_string, 'end' )
    ENDIF

 END SUBROUTINE usm_green_heat_model

!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Parin for &usm_par for urban surface model
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE usm_parin

    IMPLICIT NONE

    CHARACTER(LEN=100)  ::  line  !< string containing current line of file PARIN

    INTEGER(iwp) ::  io_status    !< status after reading the namelist file

    LOGICAL ::  switch_off_module = .FALSE.  !< local namelist parameter to switch off the module
                                             !< although the respective module namelist appears in
                                             !< the namelist file

    NAMELIST /urban_surface_parameters/                                                            &
                        building_type,                                                             &
                        roof_category,                                                             &
                        roof_inner_temperature,                                                    &
                        roughness_concrete,                                                        &
                        switch_off_module,                                                         &
                        usm_wall_mod,                                                              &
                        wall_category,                                                             &
                        wall_inner_temperature,                                                    &
                        window_inner_temperature


!
!-- Move to the beginning of the namelist file and try to find and read the namelist.
    REWIND( 11 )
    READ( 11, urban_surface_parameters, IOSTAT=io_status )

!
!-- Action depending on the READ status
    IF ( io_status == 0 )  THEN
!
!--    urban_surface_parameters namelist was found and read correctly. Set flag that indicates that
!--    the urban surface model is switched on.
       IF ( .NOT. switch_off_module )  urban_surface = .TRUE.

    ELSEIF ( io_status > 0 )  THEN
!
!--    urban_surface_parameters namelist was found but contained errors. Print an error message
!--    including the line that caused the problem.
       BACKSPACE( 11 )
       READ( 11 , '(A)' ) line
       CALL parin_fail_message( 'urban_surface_parameters', line )

    ENDIF

 END SUBROUTINE usm_parin


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Read module-specific local restart data arrays (Fortran binary format).
!> Soubroutine reads t_surf and t_wall.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE usm_rrd_local_ftn( k, nxlf, nxlc, nxl_on_file, nxrf, nxr_on_file, nynf, nyn_on_file,   &
                               nysf, nysc, nys_on_file, found )


    USE control_parameters,                                                                        &
        ONLY: length,                                                                              &
              restart_string

    IMPLICIT NONE

    INTEGER(iwp) ::  k                 !< running index over previous input files covering current local domain
    INTEGER(iwp) ::  nxlc              !< index of left boundary on current subdomain
    INTEGER(iwp) ::  nxlf              !< index of left boundary on former subdomain
    INTEGER(iwp) ::  nxl_on_file       !< index of left boundary on former local domain
    INTEGER(iwp) ::  nxrf              !< index of right boundary on former subdomain
    INTEGER(iwp) ::  nxr_on_file       !< index of right boundary on former local domain
    INTEGER(iwp) ::  nynf              !< index of north boundary on former subdomain
    INTEGER(iwp) ::  nyn_on_file       !< index of north boundary on former local domain
    INTEGER(iwp) ::  nysc              !< index of south boundary on current subdomain
    INTEGER(iwp) ::  nysf              !< index of south boundary on former subdomain
    INTEGER(iwp) ::  nys_on_file       !< index of south boundary on former local domain
    INTEGER(iwp) ::  ns_on_file_usm    !< number of surface elements (urban type) on file
!
!-- Note, the save attribute in the following array declaration is necessary, in order to keep the
!-- number of urban surface elements on file during rrd_local calls.
    INTEGER(iwp), DIMENSION(:,:), ALLOCATABLE, SAVE ::  end_index_on_file    !<
    INTEGER(iwp), DIMENSION(:,:), ALLOCATABLE, SAVE ::  start_index_on_file  !<

    LOGICAL, INTENT(OUT)  ::  found  !<

    TYPE( surf_type_1d_usm ), SAVE ::  tmp_surf   !< temporary variable to read surface data
    TYPE( surf_type_2d_usm ), SAVE ::  tmp_wall   !< temporary variable to read wall data

    found = .TRUE.

    SELECT CASE ( restart_string(1:length) )

       CASE ( 'ns_on_file_usm')
          IF ( k == 1 )  THEN
             READ ( 13 ) ns_on_file_usm
!
!--          In case of changing mpi topology, this routine could be called more than once.
!--          Hence, arrays need to be deallocated before allocated again.
             IF ( ALLOCATED( tmp_surf%val ) )  DEALLOCATE( tmp_surf%val )
             IF ( ALLOCATED( tmp_wall%val ) )  DEALLOCATE( tmp_wall%val )

!
!--          Allocate temporary arrays for reading data on file. Note, the size of allocated surface
!--          elements do not necessarily need to match the size of present surface elements on
!--          current processor, as the number of processors between restarts can change.
             ALLOCATE( tmp_surf%val(1:ns_on_file_usm) )
             ALLOCATE( tmp_wall%val(nzb_wall:nzt_wall+1,1:ns_on_file_usm) )
          ENDIF

       CASE ( 'usm_start_index' )
          IF ( k == 1 )  THEN

             IF ( ALLOCATED( start_index_on_file ) )  DEALLOCATE( start_index_on_file )

             ALLOCATE ( start_index_on_file(nys_on_file:nyn_on_file,nxl_on_file:nxr_on_file) )

             READ ( 13 )  start_index_on_file

          ENDIF

       CASE ( 'usm_end_index' )
          IF ( k == 1 )  THEN

             IF ( ALLOCATED( end_index_on_file ) )  DEALLOCATE( end_index_on_file )

             ALLOCATE ( end_index_on_file(nys_on_file:nyn_on_file,nxl_on_file:nxr_on_file) )

             READ ( 13 )  end_index_on_file

          ENDIF

       CASE ( 't_surf_wall' )
          IF ( k == 1 )  THEN
             IF ( .NOT.  ALLOCATED( t_surf_wall%val ) )  ALLOCATE( t_surf_wall%val(1:surf_usm%ns) )
             READ ( 13 )  tmp_surf%val
          ENDIF
          CALL surface_restore_elements( t_surf_wall%val, tmp_surf%val,                            &
                                         surf_usm%start_index, start_index_on_file,                &
                                         end_index_on_file, nxlc, nysc, nxlf, nxrf, nysf, nynf,    &
                                         nys_on_file, nyn_on_file, nxl_on_file, nxr_on_file )

      CASE ( 't_surf_window' )
          IF ( k == 1 )  THEN
             IF ( .NOT.  ALLOCATED( t_surf_window%val ) )                                          &
                ALLOCATE( t_surf_window%val(1:surf_usm%ns) )
             READ ( 13 )  tmp_surf%val
          ENDIF
          CALL surface_restore_elements( t_surf_window%val, tmp_surf%val,                          &
                                         surf_usm%start_index, start_index_on_file,                &
                                         end_index_on_file, nxlc, nysc, nxlf, nxrf, nysf, nynf,    &
                                         nys_on_file, nyn_on_file, nxl_on_file,nxr_on_file )

       CASE ( 't_surf_green' )
          IF ( k == 1 )  THEN
             IF ( .NOT.  ALLOCATED( t_surf_green%val ) )                                           &
                ALLOCATE( t_surf_green%val(1:surf_usm%ns) )
             READ ( 13 )  tmp_surf%val
          ENDIF
          CALL surface_restore_elements( t_surf_green%val, tmp_surf%val,                           &
                                         surf_usm%start_index, start_index_on_file,                &
                                         end_index_on_file, nxlc, nysc, nxlf, nxrf, nysf, nynf,    &
                                         nys_on_file, nyn_on_file, nxl_on_file, nxr_on_file )

       CASE ( 'm_liq_usm' )
          IF ( k == 1 )  THEN
             IF ( .NOT.  ALLOCATED( m_liq_usm%val ) )  ALLOCATE( m_liq_usm%val(1:surf_usm%ns) )
             READ ( 13 )  tmp_surf%val
          ENDIF
          CALL surface_restore_elements( m_liq_usm%val, tmp_surf%val,                              &
                                         surf_usm%start_index, start_index_on_file,                &
                                         end_index_on_file, nxlc, nysc, nxlf, nxrf, nysf, nynf,    &
                                         nys_on_file, nyn_on_file, nxl_on_file,nxr_on_file )

       CASE ( 'swc' )
          IF ( k == 1 )  THEN
             IF ( .NOT.  ALLOCATED( swc%val ) )                                                    &
                ALLOCATE( swc%val(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
             READ ( 13 )  tmp_wall%val
          ENDIF
          CALL surface_restore_elements( swc%val, tmp_wall%val,                                    &
                                         surf_usm%start_index, start_index_on_file,                &
                                         end_index_on_file, nxlc, nysc, nxlf, nxrf, nysf, nynf,    &
                                         nys_on_file, nyn_on_file, nxl_on_file, nxr_on_file )

       CASE ( 't_wall' )
          IF ( k == 1 )  THEN
             IF ( .NOT.  ALLOCATED( t_wall%val ) )                                                 &
                ALLOCATE( t_wall%val(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
             READ ( 13 )  tmp_wall%val
          ENDIF
          CALL surface_restore_elements( t_wall%val, tmp_wall%val,                                 &
                                         surf_usm%start_index, start_index_on_file,                &
                                         end_index_on_file, nxlc, nysc, nxlf, nxrf, nysf, nynf,    &
                                         nys_on_file, nyn_on_file, nxl_on_file,nxr_on_file )

       CASE ( 't_window' )
          IF ( k == 1 )  THEN
             IF ( .NOT.  ALLOCATED( t_window%val ) )                                               &
                ALLOCATE( t_window%val(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
             READ ( 13 )  tmp_wall%val
          ENDIF
          CALL surface_restore_elements( t_window%val, tmp_wall%val,                               &
                                         surf_usm%start_index, start_index_on_file,                &
                                         end_index_on_file, nxlc, nysc, nxlf, nxrf, nysf, nynf,    &
                                         nys_on_file, nyn_on_file, nxl_on_file, nxr_on_file )

       CASE ( 't_green' )
          IF ( k == 1 )  THEN
             IF ( .NOT.  ALLOCATED( t_green%val ) )                                                &
                ALLOCATE( t_green%val(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
             READ ( 13 )  tmp_wall%val
          ENDIF
          CALL surface_restore_elements( t_green%val, tmp_wall%val,                                &
                                         surf_usm%start_index, start_index_on_file,                &
                                         end_index_on_file, nxlc, nysc, nxlf, nxrf, nysf, nynf,    &
                                         nys_on_file, nyn_on_file, nxl_on_file,nxr_on_file )

       CASE DEFAULT

          found = .FALSE.

    END SELECT

 END SUBROUTINE usm_rrd_local_ftn


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Read module-specific local restart data arrays (MPI-IO).
!> Soubroutine reads t_surf and t_wall.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE usm_rrd_local_mpi

    INTEGER(idp), DIMENSION(nys:nyn,nxl:nxr) ::  global_end_index
    INTEGER(idp), DIMENSION(nys:nyn,nxl:nxr) ::  global_start_index

    LOGICAL ::  array_found  !<
    LOGICAL ::  data_to_read !< dummy variable

!-- At the moment reading of surface data in combination with cyclic fill is not realized,
!-- so that this is skipped for the moment.
    IF ( cyclic_fill_initialization )  RETURN

    CALL rd_mpi_io_check_array( 'usm_global_start', found = array_found )
    IF ( array_found )  CALL rrd_mpi_io( 'usm_global_start', global_start_index )

    CALL rd_mpi_io_check_array( 'usm_global_end', found = array_found )
    IF ( array_found )  CALL rrd_mpi_io( 'usm_global_end', global_end_index )
!
!-- Check if data input for surface-type variables is required. Note, only invoke routine if USM
!-- surface restart data is on file. In case of cyclic fill initialization this is not necessarily
!-- guaranteed. To check this use the array_found control flag.
    IF ( array_found )  THEN
       CALL rd_mpi_io_surface_filetypes( surf_usm%start_index, surf_usm%end_index,                 &
                                         data_to_read, global_start_index, global_end_index )
    ELSE
       data_to_read = .FALSE.
    ENDIF

    IF ( data_to_read )  THEN
       CALL rd_mpi_io_check_array( 't_surf_wall', found = array_found )
       IF ( array_found )  THEN
          IF ( .NOT. ALLOCATED( t_surf_wall%val ) )  ALLOCATE( t_surf_wall%val(1:surf_usm%ns) )
          CALL rrd_mpi_io_surface( 't_surf_wall', t_surf_wall%val )
       ENDIF

       CALL rd_mpi_io_check_array( 't_surf_window', found = array_found )
       IF ( array_found )  THEN
          IF ( .NOT. ALLOCATED( t_surf_window%val ) )  ALLOCATE( t_surf_window%val(1:surf_usm%ns) )
          CALL rrd_mpi_io_surface( 't_surf_window', t_surf_window%val )
       ENDIF

       CALL rd_mpi_io_check_array( 't_surf_green', found = array_found )
       IF ( array_found )  THEN
          IF ( .NOT. ALLOCATED( t_surf_green%val ) )  ALLOCATE( t_surf_green%val(1:surf_usm%ns) )
          CALL rrd_mpi_io_surface( 't_surf_green', t_surf_green%val )
       ENDIF

       CALL rd_mpi_io_check_array( 'm_liq_usm', found = array_found )
       IF ( array_found )  THEN
          IF ( .NOT. ALLOCATED( m_liq_usm%val ) )  ALLOCATE( m_liq_usm%val(1:surf_usm%ns) )
          CALL rrd_mpi_io_surface( 'm_liq_usm', m_liq_usm%val )
       ENDIF

    ENDIF

    CALL rd_mpi_io_check_array( 'usm_global_start_2', found = array_found )
    IF ( array_found )  CALL rrd_mpi_io( 'usm_global_start_2', global_start_index )

    CALL rd_mpi_io_check_array( 'usm_global_end_2', found = array_found )
    IF ( array_found )  CALL rrd_mpi_io( 'usm_global_end_2', global_end_index )

    IF ( array_found )  THEN
       CALL rd_mpi_io_surface_filetypes( surf_usm%start_index, surf_usm%end_index,                 &
                                         data_to_read, global_start_index, global_end_index )
    ELSE
       data_to_read = .FALSE.
    ENDIF

    IF ( data_to_read )  THEN
       CALL rd_mpi_io_check_array( 'swc', found = array_found )
       IF ( array_found )  THEN
          IF ( .NOT. ALLOCATED( swc%val ) )  ALLOCATE( swc%val(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
          CALL rrd_mpi_io_surface( 'swc', swc%val )
       ENDIF

       CALL rd_mpi_io_check_array( 't_wall', found = array_found )
       IF ( array_found )  THEN
          IF ( .NOT. ALLOCATED( t_wall%val ) )                                                     &
             ALLOCATE( t_wall%val(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
          CALL rrd_mpi_io_surface( 't_wall', t_wall%val )
       ENDIF

       CALL rd_mpi_io_check_array( 't_window', found = array_found )
          IF ( array_found )  THEN
          IF ( .NOT. ALLOCATED( t_window%val ) )                                                   &
             ALLOCATE( t_window%val(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
          CALL rrd_mpi_io_surface( 't_window', t_window%val )
       ENDIF

       CALL rd_mpi_io_check_array( 't_green', found = array_found )
       IF ( array_found )  THEN
          IF ( .NOT. ALLOCATED( t_green%val ) )                                                    &
             ALLOCATE( t_green%val(nzb_wall:nzt_wall+1,1:surf_usm%ns) )
          CALL rrd_mpi_io_surface( 't_green', t_green%val )
       ENDIF
    ENDIF

 END SUBROUTINE usm_rrd_local_mpi


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Solver for the energy balance at the ground/roof/wall surface. It follows the basic ideas and
!> structure of lsm_energy_balance with many simplifications and adjustments.
!> TODO better description
!> No calculation of window surface temperatures during spinup to increase maximum possible timstep
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE usm_energy_balance( during_spinup )

    LOGICAL ::  during_spinup  !< flag indicating soil/wall spinup phase

    CALL usm_surface_energy_balance( during_spinup )

    CALL usm_green_heat_model                      !< usm_green_heat_model has not been vectorized yet

    CALL usm_wall_heat_model( during_spinup )

 END SUBROUTINE usm_energy_balance


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Solver for the energy balance at the ground/roof/wall surface. It follows the basic ideas and
!> structure of lsm_energy_balance with many simplifications and adjustments.
!> TODO better description
!> No calculation of window surface temperatures during spinup to increase maximum possible timstep
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE usm_surface_energy_balance( during_spinup )

    IMPLICIT NONE

    LOGICAL, INTENT(IN)               ::  during_spinup  !< flag indicating soil/wall spinup phase
    LOGICAL                           ::  runge_l        !< dummy flag to indicate RK timestepping scheme

    LOGICAL, DIMENSION(1:surf_usm%ns) ::  horizontal                !< flag indicating horizontal surfaces
    LOGICAL, DIMENSION(1:surf_usm%ns) ::  force_radiation_call_l_v  !< flag to gather information if radiation need to be called

    INTEGER(iwp) ::  i              !< grid index for reference atmosphere cell, x-direction
    INTEGER(iwp) ::  j              !< grid index for reference atmosphere cell, y-direction
    INTEGER(iwp) ::  k              !< grid index for reference atmosphere cell, z-direction
    INTEGER(iwp) ::  kk             !< loop index for depth of vegetation layer
    INTEGER(iwp) ::  m              !< running index for surface elements
    INTEGER(iwp) ::  i_off          !< offset to determine index of surface element, seen from atmospheric grid point, for x
    INTEGER(iwp) ::  j_off          !< offset to determine index of surface element, seen from atmospheric grid point, for y
    INTEGER(iwp) ::  k_off          !< offset to determine index of surface element, seen from atmospheric grid point, for z


    REAL(wp) ::  coef_1                  !< first coeficient for prognostic equation
    REAL(wp) ::  coef_2                  !< second  coeficient for prognostic equation
    REAL(wp) ::  e                       !< water vapour pressure
    REAL(wp) ::  e_s                     !< water vapour saturation pressure
    REAL(wp) ::  f1                      !< resistance correction term 1
    REAL(wp) ::  f3                      !< resistance correction term 3
    REAL(wp) ::  m_max_depth = 0.0002_wp !< Maximum capacity of the water reservoir (m)
    REAL(wp) ::  stend_wall              !< tendency for wall surfaces
    REAL(wp) ::  stend_window            !< tendency for window surfaces
    REAL(wp) ::  stend_green             !< tendency for green surfaces
    REAL(wp) ::  tend                    !< tendency
    REAL(wp) ::  ueff                    !< limited near-surface wind speed - used for calculation of resistance


    REAL(wp), DIMENSION(1:surf_usm%ns) ::  coef_green_1  !< first coeficient for prognostic green wall equation
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  coef_green_2  !< second  coeficient for prognostic green wall equation
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  coef_window_1 !< first coeficient for prognostic window equation
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  coef_window_2 !< second  coeficient for prognostic window equation
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  dq_s_dt       !< derivate of q_s with respect to T
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  drho_l_lv     !< frequently used parameter for green layers
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  e_s_dt        !< derivate of e_s with respect to T
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  f_qsws        !< factor for qsws
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  f_qsws_veg    !< factor for qsws_veg
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  f_qsws_liq    !< factor for qsws_liq
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  f2            !< resistance correction term 2
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  frac_green    !< green fraction, used to restore original values during spinup
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  frac_win      !< window fraction, used to restore original values during spinup
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  frac_wall     !< wall fraction, used to restore original values during spinup
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  f_shf         !< factor for shf_eb
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  f_shf_green   !< factor for shf_eb green wall
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  f_shf_window  !< factor for shf_eb window
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  m_liq_max     !< maxmimum value of the liq. water reservoir
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  m_total       !< total soil moisture content
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  qv1           !< specific humidity at first grid level
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  q_s           !< saturation specific humidity
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  rho_cp        !< rho_wall_surface * c_p
    REAL(wp), DIMENSION(1:surf_usm%ns) ::  rho_lv        !< rho_wall_surface * l_v


    TYPE(surf_type), POINTER ::  surf              !< surface-date type variable


    IF ( debug_output_timestep )  THEN
       WRITE( debug_string, * ) 'usm_surface_energy_balance_vector:', during_spinup
       CALL debug_message( debug_string, 'start' )
    ENDIF

    surf  => surf_usm

    runge_l = (timestep_scheme(1:5) == 'runge')

    force_radiation_call_l_v = .FALSE.
!
!-- Set control flags
    IF ( surf%ns > 0 )  horizontal = ( surf%upward(1:surf%ns) .OR. surf%downward(1:surf%ns) )
!
!-- During spinup set green and window fraction to zero and restore at the end of the loop.
    IF ( during_spinup )  THEN
       frac_win   = 0.0_wp
       frac_wall  = 1.0_wp
       frac_green = 0.0_wp
    ELSE
       !$OMP PARALLEL DO PRIVATE (m) SCHEDULE (STATIC)
       DO  m = 1, surf%ns
          frac_win(m)   = surf%frac(m,ind_wat_win)
          frac_wall(m)  = surf%frac(m,ind_veg_wall)
          frac_green(m) = surf%frac(m,ind_pav_green)
       ENDDO
    ENDIF
!
!-- Precalculate frequently used parameters such as rho_cp and qv1.
    !$OMP PARALLEL DO PRIVATE (m, i, j, k) SCHEDULE (STATIC)
    DO  m = 1, surf%ns
!
!--    Get indices of respective grid point.
       i = surf%i(m)
       j = surf%j(m)
       k = surf%k(m)

       rho_cp(m)  = c_p * hyp(k) / ( r_d * surf%pt1(m) * exner(k) )

       IF ( frac_green(m) > 0.0_wp )  THEN
          rho_lv(m)    = rho_cp(m) / c_p * l_v
          drho_l_lv(m) = 1.0_wp / ( rho_l * l_v )
       ENDIF

       IF ( humidity )  THEN
          qv1(m) = q(k,j,i)
       ELSE
          qv1(m) = 0.0_wp
       ENDIF
    ENDDO

!
!-- Calculate aerodyamic resistance.
    !$OMP PARALLEL DO PRIVATE (m, i, j, k, ueff) SCHEDULE (STATIC)
    DO  m = 1, surf%ns
       IF ( surf%upward(m) )  THEN
!
!--       Calculation for horizontally upward facing surfaces follows LSM formulation.
!--       pt, us, ts are not available for the prognostic time step, data from the
!--       last time step is used here.
          surf%r_a(m) = ( surf%pt1(m) - surf%pt_surface(m) ) /                                     &
                        ( surf%ts(m) * surf%us(m) + 1.0E-20_wp )
       ELSE
!
!--       Get indices of respective grid point.
          i = surf%i(m)
          j = surf%j(m)
          k = surf%k(m)
!
!--       Calculation of r_a for vertical and downward facing horizontal surfaces
!--
!--       Heat transfer coefficient for forced convection along vertical walls follows formulation
!--       in TUF3d model (Krayenhoff & Voogt, 2006)
!--
!--       H = httc (Tsfc - Tair)
!--       httc = rw * (11.8 + 4.2 * Ueff) - 4.0
!--
!--            rw: Wall patch roughness relative to 1.0 for concrete
!--            Ueff: Effective wind speed
!--            - 4.0 is a reduction of Rowley et al (1930) formulation based on
!--            Cole and Sturrock (1977)
!--
!--            Ucan: Canyon wind speed
!--            wstar: Convective velocity
!--            Qs: Surface heat flux
!--            zH: Height of the convective layer
!--            wstar = (g/Tcan*Qs*zH)**(1./3.)
!--       Effective velocity components must always be defined at scalar grid point. The wall
!--       normal component is obtained by simple linear interpolation. (An alternative would be an
!--       logarithmic interpolation.) Parameter roughness_concrete (default value = 0.001) is used
!--       to calculation of roughness relative to concrete. Note, wind velocity is limited
!--       to avoid division by zero. The nominator can become <= 0.0 for values z0 < 3*10E-4.
          ueff = MAX( SQRT( ( ( u(k,j,i) + u(k,j,i+1) ) * 0.5_wp )**2 +                            &
                            ( ( v(k,j,i) + v(k,j+1,i) ) * 0.5_wp )**2 +                            &
                            ( ( w(k,j,i) + w(k-1,j,i) ) * 0.5_wp )**2 ),                           &
                      ( ( 4.0_wp + 0.1_wp ) / ( surf%z0(m) * d_roughness_concrete ) - 11.8_wp )    &
                      / 4.2_wp                                                                     &
                    )

          surf%r_a(m) = rho_cp(m) / ( surf%z0(m) * d_roughness_concrete *                          &
                        ( 11.8_wp + 4.2_wp * ueff ) - 4.0_wp  )
       ENDIF
    ENDDO

    IF ( surf%ns > 0 )  THEN
!
!--    Make sure that the resistance does not drop to zero and does not exceed a maxmium value in
!--    case of zero velocities.
       WHERE ( surf%r_a(1:surf%ns) < 1.0_wp   )  surf%r_a(1:surf%ns) = 1.0_wp
       WHERE ( surf%r_a(1:surf%ns) > 300.0_wp )  surf%r_a(1:surf%ns) = 300.0_wp
!
!--    Aeorodynamical resistance for the window and green fractions are set to the same value.
       surf%r_a_window(1:surf%ns) = surf%r_a(1:surf%ns)
       surf%r_a_green(1:surf%ns)  = surf%r_a(1:surf%ns)
!
!--    Factor for shf_eb.
       f_shf(1:surf%ns)        = rho_cp(1:surf%ns) / surf%r_a(1:surf%ns)
       f_shf_window(1:surf%ns) = rho_cp(1:surf%ns) / surf%r_a_window(1:surf%ns)
       f_shf_green(1:surf%ns)  = rho_cp(1:surf%ns) / surf%r_a_green(1:surf%ns)

    ENDIF

    !$OMP PARALLEL DO PRIVATE (m) SCHEDULE (STATIC)
    DO  m = 1, surf%ns
       IF ( frac_green(m) > 0.0_wp )  THEN
          IF ( surf%upward(m) ) THEN
!
!--          For upward-facing surfaces, compute soil moisture content. This is required for
!--          correction factor f2.
             m_total(m) = 0.0_wp
             DO  kk = nzb_wall, nzt_wall+1
                m_total(m) = m_total(m) + rootfr%val(nzb_wall,m) *                                 &
                                          MAX( swc%val(nzb_wall,m), wilt%val(nzb_wall,m) )
             ENDDO
!
!--          f2: Correction for soil moisture availability to plants (the integrated soil moisture
!--          must thus be considered here). f2 = 0 for very dry soils.
             IF ( m_total(m) > wilt%val(nzb_wall,m)  .AND.  m_total(m) < fc%val(nzb_wall,m) )  THEN
                f2(m) = ( m_total(m)         - wilt%val(nzb_wall,m) ) /                            &
                        ( fc%val(nzb_wall,m) - wilt%val(nzb_wall,m) )
             ELSEIF ( m_total(m) >= fc%val(nzb_wall,m) )  THEN
                f2(m) = 1.0_wp
             ELSE
                f2(m) = 1.0E-20_wp
             ENDIF
          ELSE
!
!--          f2 = 1 for vertical surfaces.
             f2(m) = 1.0_wp
          ENDIF
       ENDIF
    ENDDO

    !$OMP PARALLEL DO PRIVATE (m, f1, f3, e_s, e, coef_1, coef_2) SCHEDULE (STATIC)
    DO  m = 1, surf%ns

       IF ( frac_green(m) > 0.0_wp )  THEN
!
!--       Adapted from LSM:
!--       Second step: calculate canopy resistance r_canopy. f1-f3 here are defined as 1/f1-f3
!--       as in ECMWF documentation.
!--       f1: Correction for incoming shortwave radiation (stomata close at night).
          f1 = MIN( 1.0_wp, ( 0.004_wp * surf%rad_sw_in(m) + 0.05_wp ) /                           &
                            ( 0.81_wp * ( 0.004_wp * surf%rad_sw_in(m) + 1.0_wp ) ) )
!
!--       Calculate water vapour pressure at saturation.
          e_s = 0.01_wp * magnus_tl( t_surf_green%val(m) )
!
!--       f3: Correction for vapour pressure deficit.
          IF ( surf%g_d(m) /= 0.0_wp )  THEN
!
!--          Calculate vapour pressure.
             e  = qv1(m) * surface_pressure / ( qv1(m) + 0.622_wp )
             f3 = EXP ( - surf%g_d(m) * ( e_s - e ) )
          ELSE
             f3 = 1.0_wp
          ENDIF
!
!--       Calculate canopy resistance. In case that c_veg is 0 (bare soils), this calculation is
!--       obsolete, as r_canopy is not used below.
!--       To do: check for very dry soil -> r_canopy goes to infinity.
          surf%r_canopy(m) = surf%r_canopy_min(m) / ( surf%lai(m) * f1 * f2(m) * f3 + 1.0E-20_wp )
!
!--       Calculate saturation specific humidity.
          q_s(m) = 0.622_wp * e_s / ( surface_pressure - e_s )
!
!--       In case of dewfall, set evapotranspiration to zero.
!--       All super-saturated water is then removed from the air.
          IF ( humidity  .AND.  q_s(m) <= qv1(m) )  THEN
             surf%r_canopy(m) = 0.0_wp
          ENDIF

          IF ( surf%upward(m) )  THEN
!
!--          Calculate the maximum possible liquid water amount on plants and bare surface. For
!--          vegetated surfaces, a maximum depth of 0.2 mm is assumed, while paved surfaces might
!--          hold up 1 mm of water. The liquid water fraction for paved surfaces is calculated after
!--          Noilhan & Planton (1989), while the ECMWF formulation is used for vegetated surfaces
!--          and bare soils.
             m_liq_max(m) = m_max_depth * ( surf%lai(m) )
             surf%c_liq(m) = MIN( 1.0_wp, ( m_liq_usm%val(m) / m_liq_max(m) )**0.67 )

!
!--          Calculate coefficients for the total evapotranspiration.
!--          In case of water surface, set vegetation and soil fluxes to zero.
!--          For pavements, only evaporation of liquid water is possible.
             f_qsws_veg(m)  = rho_lv(m) * ( 1.0_wp - surf%c_liq(m) ) /                             &
                              ( surf%r_a_green(m) + surf%r_canopy(m) )
             f_qsws_liq(m)  = rho_lv(m) * surf%c_liq(m) / surf%r_a_green(m)
             f_qsws(m) = f_qsws_veg(m) + f_qsws_liq(m)
          ELSE
             f_qsws_veg(m)  = rho_lv(m) * ( 1.0_wp - 0.0_wp ) /                                    &
                              ( surf%r_a_green(m) + surf%r_canopy(m) )
             f_qsws_liq(m)  = 0.0_wp ! rho_lv(m) * surf%c_liq(m) / surf%r_a_green(m)
             f_qsws(m) = f_qsws_veg(m) + f_qsws_liq(m)
          ENDIF
!
!--       Calculate derivative of q_s for Taylor series expansion.
          e_s_dt(m) = e_s * ( 17.269_wp / ( t_surf_green%val(m) - 35.86_wp )                       &
                            - 17.269_wp  * ( t_surf_green%val(m) - degc_to_k ) /                   &
                                           ( t_surf_green%val(m) - 35.86_wp )**2                   &
                            )
          dq_s_dt(m) = 0.622_wp * e_s_dt(m) / ( surface_pressure - e_s_dt(m) )
       ENDIF
    ENDDO
!
!-- Add LW up so that it can be removed in prognostic equation.
    IF ( surf%ns > 0 )  THEN
       surf%rad_net_l(1:surf%ns) = surf%rad_sw_in(1:surf%ns)  - surf%rad_sw_out(1:surf%ns) +       &
                                   surf%rad_lw_in(1:surf%ns)  - surf%rad_lw_out(1:surf%ns)
    ENDIF

!
!-- Compute coef_window_1 and coef_window_2.
    coef_window_1 = 0.0_wp
    coef_window_2 = 0.0_wp
    IF ( .NOT. during_spinup )  THEN
       !$OMP PARALLEL DO PRIVATE (k, m) SCHEDULE (STATIC)
       DO  m = 1, surf%ns
          IF ( frac_win(m) > 0.0_wp )  THEN
!
!--          Get k index of respective grid point.
             k = surf%k(m)
             coef_window_1(m) = surf%rad_net_l(m) +  ( 3.0_wp + 1.0_wp ) *                         &
                                surf%emissivity(m,ind_wat_win) * sigma_sb *                        &
                                t_surf_window%val(m)**4 + f_shf_window(m) * surf%pt1(m) +          &
                                surf%lambda_surf_window(m) * t_window%val(nzb_wall,m)

             coef_window_2(m) = 4.0_wp * surf%emissivity(m,ind_wat_win) * sigma_sb *               &
                                t_surf_window%val(m)**3 + surf%lambda_surf_window(m) +             &
                                f_shf_window(m) / exner(k)
          ENDIF
       ENDDO
    ENDIF
!
!-- Compute coef_green_1 and coef_green_2.
    !$OMP PARALLEL DO PRIVATE (k, m) SCHEDULE (STATIC)
    DO  m = 1, surf%ns
!
!--    Get k index of respective grid point.
       k = surf%k(m)
       IF ( humidity  .AND.  frac_green(m) > 0.0_wp )  THEN
          coef_green_1(m) = surf%rad_net_l(m) + ( 3.0_wp + 1.0_wp ) *                              &
                         surf%emissivity(m,ind_pav_green) * sigma_sb *                             &
                         t_surf_green%val(m)**4 + f_shf_green(m) * surf%pt1(m) +                   &
                         f_qsws(m) * ( qv1(m) - q_s(m) + dq_s_dt(m) * t_surf_green%val(m) ) +      &
                         surf%lambda_surf_green(m) * t_green%val(nzb_wall,m)

          coef_green_2(m) = 4.0_wp * surf%emissivity(m,ind_pav_green) * sigma_sb *                 &
                         t_surf_green%val(m)**3 + f_qsws(m) * dq_s_dt(m) +                         &
                         surf%lambda_surf_green(m) + f_shf_green(m) / exner(k)
       ELSE
          coef_green_1(m) = surf%rad_net_l(m) + ( 3.0_wp + 1.0_wp ) *                              &
                            surf%emissivity(m,ind_pav_green) * sigma_sb * t_surf_green%val(m)**4 + &
                            f_shf_green(m) * surf%pt1(m) + surf%lambda_surf_green(m) *             &
                            t_green%val(nzb_wall,m)
          coef_green_2(m) = 4.0_wp * surf%emissivity(m,ind_pav_green) * sigma_sb *                 &
                            t_surf_green%val(m)**3 + surf%lambda_surf_green(m) +                   &
                            f_shf_green(m) / exner(k)
       ENDIF
    ENDDO

    !$OMP PARALLEL DO PRIVATE (m, coef_1, coef_2, stend_wall, stend_window,                        &
    !$OMP&                     stend_green ) SCHEDULE (STATIC)
    DO  m = 1, surf%ns
!
!--    Get k index of respective grid point.
       k = surf%k(m)
!
!--    Numerator of the prognostic equation.
!--    Todo: Adjust to tile approach. So far, emissivity for wall (element 0) is used
!--    Rem: Coef +1 corresponds to -lwout included in calculation of radnet_l.
       coef_1 = surf%rad_net_l(m) + ( 3.0_wp + 1.0_wp ) *                                          &
                surf%emissivity(m,ind_veg_wall) * sigma_sb * t_surf_wall%val(m)**4 +               &
                f_shf(m) * surf%pt1(m) + surf%lambda_surf(m) * t_wall%val(nzb_wall,m)

!
!--    Denominator of the prognostic equation.
       coef_2 = 4.0_wp * surf%emissivity(m,ind_veg_wall) * sigma_sb * t_surf_wall%val(m)**3 +      &
                surf%lambda_surf(m) + f_shf(m) / exner(k)
!
!--    Implicit solution when the surface layer has no heat capacity, otherwise use RK3 scheme.
       t_surf_wall_p%val(m) = ( coef_1 * dt_3d * tsc(2) + surf%c_surface(m) *                      &
                              t_surf_wall%val(m) ) / ( surf%c_surface(m) + coef_2 * dt_3d * tsc(2) )

       IF ( .NOT. during_spinup  .AND.  frac_win(m) > 0.0_wp )  THEN
          t_surf_window_p%val(m) = ( coef_window_1(m) * dt_3d * tsc(2) +                           &
                                   surf%c_surface_window(m) * t_surf_window%val(m) ) /             &
                                   ( surf%c_surface_window(m) + coef_window_2(m) * dt_3d * tsc(2) )
       ENDIF
       t_surf_green_p%val(m) = ( coef_green_1(m) * dt_3d * tsc(2) +                                &
                               surf%c_surface_green(m) * t_surf_green%val(m) ) /                   &
                               ( surf%c_surface_green(m) + coef_green_2(m) * dt_3d * tsc(2) )

!
!--    Add RK3 term.
       t_surf_wall_p%val(m) = t_surf_wall_p%val(m)     + dt_3d * tsc(3) *                          &
                                                         surf%tt_surface_wall_m(m)
       t_surf_window_p%val(m) = t_surf_window_p%val(m) + dt_3d * tsc(3) *                          &
                                                         surf%tt_surface_window_m(m)
       t_surf_green_p%val(m) = t_surf_green_p%val(m)   + dt_3d * tsc(3) *                          &
                                                         surf%tt_surface_green_m(m)

!
!--    Store surface temperature on pt_surface. Further, in case humidity is used, store also
!--    vpt_surface, which is, due to the lack of moisture on roofs, simply assumed to be the
!--    surface temperature.
       surf%pt_surface(m) = ( frac_wall(m)  * t_surf_wall_p%val(m)                                 &
                            + frac_win(m)   * t_surf_window_p%val(m)                               &
                            + frac_green(m) * t_surf_green_p%val(m)                                &
                             ) / exner(k)

!
!--    Following line is actually not fully correct. In order to overcome this, a q_surface
!--    would be needed, calculated according to the q_surface in the LSM, where it is assumed
!--    that the skin layer is saturated. However, it is not clear whether this makes much sence
!--    in case of walls and windows. Probably only for green surfaces.
       IF ( humidity )  surf%vpt_surface(m) = surf%pt_surface(m)
!
!--    Calculate true tendency
       stend_wall = ( t_surf_wall_p%val(m) - t_surf_wall%val(m)       - dt_3d * tsc(3) *           &
                      surf%tt_surface_wall_m(m) ) / ( dt_3d  * tsc(2) )
       stend_window = ( t_surf_window_p%val(m) - t_surf_window%val(m) - dt_3d * tsc(3) *           &
                        surf%tt_surface_window_m(m) ) / ( dt_3d  * tsc(2) )
       stend_green = ( t_surf_green_p%val(m) - t_surf_green%val(m)    - dt_3d * tsc(3) *           &
                       surf%tt_surface_green_m(m) ) / ( dt_3d  * tsc(2) )
!
!--    Calculate t_surf tendencies for the next Runge-Kutta step.
       IF ( runge_l )  THEN
          IF ( intermediate_timestep_count == 1 )  THEN
             surf%tt_surface_wall_m(m)   = stend_wall
             surf%tt_surface_window_m(m) = stend_window
             surf%tt_surface_green_m(m)  = stend_green
          ELSEIF ( intermediate_timestep_count < intermediate_timestep_count_max )  THEN
             surf%tt_surface_wall_m(m) = -9.5625_wp * stend_wall +                                 &
                                          5.3125_wp * surf%tt_surface_wall_m(m)
             surf%tt_surface_window_m(m) = -9.5625_wp * stend_window +                             &
                                            5.3125_wp * surf%tt_surface_window_m(m)
             surf%tt_surface_green_m(m) = -9.5625_wp * stend_green +                               &
                                           5.3125_wp * surf%tt_surface_green_m(m)
          ENDIF
       ENDIF
    ENDDO

    !$OMP PARALLEL DO PRIVATE (m) SCHEDULE (STATIC)
    DO  m = 1, surf%ns
!
!--    In case of fast changes in the skin temperature, it is required to update the radiative
!--    fluxes in order to keep the solution stable.
       IF ( ( ( ABS( t_surf_wall_p%val(m)   - t_surf_wall%val(m) )   > 1.0_wp )   .OR.             &
              ( ABS( t_surf_green_p%val(m)  - t_surf_green%val(m) )  > 1.0_wp )   .OR.             &
              ( ABS( t_surf_window_p%val(m) - t_surf_window%val(m) ) > 1.0_wp ) )                  &
               .AND.  unscheduled_radiation_calls  )                                               &
       THEN
          force_radiation_call_l_v(m) = .TRUE.
       ENDIF
    ENDDO

    !$OMP PARALLEL DO PRIVATE (m, tend, i, i_off, j, j_off, k, k_off ) SCHEDULE (STATIC)
    DO  m = 1, surf%ns
!
!--    Index offset of surface element point with respect to adjoining atmospheric grid point.
       k_off = surf%koff(m)
       j_off = surf%joff(m)
       i_off = surf%ioff(m)
!
!--    Get indices of respective grid point.
       i = surf%i(m)
       j = surf%j(m)
       k = surf%k(m)
!
!--    Calculate new fluxes
!--    rad_net_l is never used!
       surf%rad_net_l(m) = surf%rad_net_l(m) +                                                     &
                           frac_wall(m)  * sigma_sb * surf%emissivity(m,ind_veg_wall) *            &
                                          ( t_surf_wall_p%val(m)**4 - t_surf_wall%val(m)**4 ) +    &
                           frac_win(m)   * sigma_sb * surf%emissivity(m,ind_wat_win) *             &
                                          ( t_surf_window_p%val(m)**4 - t_surf_window%val(m)**4 ) +&
                           frac_green(m) * sigma_sb * surf%emissivity(m,ind_pav_green) *           &
                                          ( t_surf_green_p%val(m)**4 - t_surf_green%val(m)**4 )

       surf%wghf_eb(m)        = surf%lambda_surf(m) *                                              &
                                ( t_surf_wall_p%val(m)   - t_wall%val(nzb_wall,m) )
       surf%wghf_eb_green(m)  = surf%lambda_surf_green(m) *                                        &
                                ( t_surf_green_p%val(m)  - t_green%val(nzb_wall,m) )
       surf%wghf_eb_window(m) = surf%lambda_surf_window(m) *                                       &
                                ( t_surf_window_p%val(m) - t_window%val(nzb_wall,m) )
!
!--    Ground/wall/roof surface heat flux.
       surf%wshf_eb(m) = - f_shf(m)        * ( surf%pt1(m) - t_surf_wall_p%val(m)   / exner(k) ) * &
                           frac_wall(m)                                                            &
                         - f_shf_window(m) * ( surf%pt1(m) - t_surf_window_p%val(m) / exner(k) ) * &
                           frac_win(m)                                                             &
                         - f_shf_green(m)  * ( surf%pt1(m) - t_surf_green_p%val(m)  / exner(k) ) * &
                           frac_green(m)
!
!--    Store kinematic surface heat fluxes for utilization in other processes diffusion_s,
!--    surface_layer_fluxes,...
       surf%shf(m) = surf%wshf_eb(m) / c_p
!
!--    If the indoor model is applied, further add waste heat from buildings to the kinematic flux.
       IF ( indoor_model  .OR.  &
            ( external_anthropogenic_heat  .AND.  ALLOCATED( surf%waste_heat ) ) )  THEN
          surf%shf(m) = surf%shf(m) + surf%waste_heat(m) / c_p  !! TODO: Check if the division by c_p is correct
       ENDIF
!
!--    Following line is necessary to remove the density from the flux. For horizontal surfaces
!--    where the heat flux is added to the vertical diffusion term, density cancels out in
!--    diffusion_s.f90. However, for vertical surfaces the density would still be included in the
!--    diffusion terms, meaning that the heat-fluxes at the walls would be overestimated by about
!--    15-20%. Please note, here in the building-surface model density is expressed by
!--    hyp(k) / ( r_d * surf%pt1(m) * exner(k) ).
       IF ( .NOT. horizontal(m) )  THEN
          surf%shf(m) = surf%shf(m) * ( r_d * surf%pt1(m) * exner(k) ) / hyp(k)
       ENDIF

       IF ( humidity  .AND.  frac_green(m) > 0.0_wp )  THEN
!
!--       Calculate true surface resistance.
          IF ( surf%upward(m) ) THEN
             surf%qsws(m) = -f_qsws(m) * ( qv1(m) - q_s(m) +                                       &
                             dq_s_dt(m) * t_surf_green%val(m) -                                    &
                             dq_s_dt(m) * t_surf_green_p%val(m) )
             surf%qsws(m) = surf%qsws(m) / l_v
             surf%qsws_veg(m) = -f_qsws_veg(m) * ( qv1(m) - q_s(m) +                               &
                                 dq_s_dt(m) * t_surf_green%val(m) -                                &
                                 dq_s_dt(m) * t_surf_green_p%val(m) )
             surf%qsws_liq(m)  = -f_qsws_liq(m) * ( qv1(m) - q_s(m) +                              &
                                  dq_s_dt(m) * t_surf_green%val(m) -                               &
                                  dq_s_dt(m) * t_surf_green_p%val(m) )
             surf%r_s(m) = -rho_lv(m) * ( qv1(m) - q_s(m) +                                        &
                            dq_s_dt(m) * t_surf_green%val(m) -                                     &
                            dq_s_dt(m) * t_surf_green_p%val(m) ) /                                 &
                            ( surf%qsws(m) + 1.0E-20 ) - surf%r_a_green(m)

             IF ( precipitation )  THEN
!
!--             Calculate change in liquid water reservoir due to dew fall or evaporation of liquid
!--             water. If precipitation is activated, add rain water to qsws_liq and qsws_soil
!--             according to the vegetation coverage. Precipitation_rate is given in mm.
!--             Add precipitation to liquid water reservoir, if possible. Otherwise, add the water
!--             to soil. In case of pavements, the exceeding water amount is implicitely removed as
!--             runoff as qsws_soil is then not used in the soil model
                IF ( m_liq_usm%val(m) /= m_liq_max(m) )  THEN
                   surf%qsws_liq(m) = surf%qsws_liq(m) + frac_green(m) * rho_l * l_v * 0.001_wp *  &
                                      prr(k+k_off,j+j_off,i+i_off) * hyrho(k+k_off)

                ENDIF
             ENDIF
!
!--          If the air is saturated, check the reservoir water level.
             IF ( surf%qsws(m) < 0.0_wp )  THEN
!
!--             Check if reservoir is full (avoid values > m_liq_max) In that case, qsws_liq goes to
!--             qsws_soil. In this case qsws_veg is zero anyway (because c_liq = 1), so that tend is
!--             zero and no further check is needed.
                IF ( m_liq_usm%val(m) == m_liq_max(m) )  THEN
                   surf%qsws_liq(m)  = 0.0_wp
                ENDIF
!
!--             In case qsws_veg becomes negative (unphysical behavior), let the water enter the
!--             liquid water reservoir as dew on the plant.
                IF ( surf%qsws_veg(m) < 0.0_wp )  THEN
                   surf%qsws_liq(m) = surf%qsws_liq(m) + surf%qsws_veg(m)
                   surf%qsws_veg(m) = 0.0_wp
                ENDIF
             ENDIF

             tend = -surf%qsws_liq(m) * drho_l_lv(m)
             m_liq_usm_p%val(m) = m_liq_usm%val(m) + dt_3d *                                       &
                                  ( tsc(2) * tend + tsc(3) * tm_liq_usm_m%val(m) )
!
!--          Check if reservoir is overfull -> reduce to maximum
!--          (conservation of water is violated here)
             m_liq_usm_p%val(m) = MIN( m_liq_usm_p%val(m), m_liq_max(m) )
!
!--          Check if reservoir is empty (avoid values < 0.0) (conservation of water is
!--          violated here).
             m_liq_usm_p%val(m) = MAX( m_liq_usm_p%val(m), 0.0_wp )
!
!--          Calculate m_liq tendencies for the next Runge-Kutta step
             IF ( runge_l )  THEN
                IF ( intermediate_timestep_count == 1 )  THEN
                   tm_liq_usm_m%val(m) = tend
                ELSEIF ( intermediate_timestep_count < intermediate_timestep_count_max )  THEN
                   tm_liq_usm_m%val(m) = -9.5625_wp * tend + 5.3125_wp * tm_liq_usm_m%val(m)
                ENDIF
             ENDIF
          ELSE
!
!--          Downward or vertical surfaces.
             surf%qsws(m) = - f_qsws(m) * ( qv1(m) - q_s(m) + dq_s_dt(m) * t_surf_green%val(m)     &
                                                   - dq_s_dt(m) * t_surf_green_p%val(m) )
             surf%qsws(m) = surf%qsws(m) / l_v
!
!--          Following line is necessary to remove the density from the flux. For horizontal
!--          surfaceswhere the heat flux is added to the vertical diffusion term, density cancels
!--          out in diffusion_s.f90. However, for vertical surfaces the density would still be
!--          included in the diffusion terms, meaning that the heat-fluxes at the walls would be
!--          overestimated by about 15-20%. Please note, here in the building-surface model density
!--          is expressed by hyp(k) / ( r_d * surf%pt1(m) * exner(k) )
             IF( .NOT. horizontal(m) )  surf%qsws(m) = surf%qsws(m) *                              &
                                                      ( r_d * surf%pt1(m) * exner(k) ) / hyp(k)

             surf%qsws_veg(m) = -f_qsws_veg(m) * ( qv1(m) - q_s(m) +                               &
                                                   dq_s_dt(m) * t_surf_green%val(m) -              &
                                                   dq_s_dt(m) * t_surf_green_p%val(m) )
             surf%r_s(m) = -rho_lv(m) * ( qv1(m) - q_s(m) +                                        &
                                          dq_s_dt(m) * t_surf_green%val(m) -                       &
                                          dq_s_dt(m) * t_surf_green_p%val(m) ) /                   &
                                        ( surf%qsws(m) + 1.0E-20_wp ) - surf%r_a_green(m)
             surf%qsws_liq(m) = 0.0_wp  ! - f_qsws_liq(m)  * ( qv1(m) - q_s + dq_s_dt(m) * t_surf_green_h(m)&
!                                                  - dq_s_dt(m) * t_surf_green_h_p(m) )
!
!--          If the air is saturated, check the reservoir water level.
             IF ( surf%qsws(m) < 0.0_wp )  THEN
!
!--             In case qsws_veg becomes negative (unphysical behavior), let the water enter the
!--             liquid water reservoir as dew on the plant.
                IF ( surf%qsws_veg(m) < 0.0_wp )  THEN
                   surf%qsws_veg(m) = 0.0_wp
                ENDIF
             ENDIF
          ENDIF
       ELSE
          surf%r_s(m) = 1.0E10_wp
       ENDIF

    ENDDO
    force_radiation_call_l = ANY( force_radiation_call_l_v )
!
!-- Calculation of force_radiation_call:
!-- Make logical OR for all processes.
!-- Force radiation call if at least one processor forces it.
    IF ( intermediate_timestep_count == intermediate_timestep_count_max-1 )  THEN
#if defined( __parallel )
       IF ( .NOT. force_radiation_call ) THEN
          IF ( collective_wait )  CALL MPI_BARRIER( comm2d, ierr )
          CALL MPI_ALLREDUCE( force_radiation_call_l, force_radiation_call,                        &
                              1, MPI_LOGICAL, MPI_LOR, comm2d, ierr )
       ENDIF
#else
       force_radiation_call = ( force_radiation_call  .OR.  force_radiation_call_l )
#endif
       force_radiation_call_l = .FALSE.
    ENDIF

    IF ( debug_output_timestep )  THEN
       WRITE( debug_string, * ) 'usm_surface_energy_balance: ', during_spinup
       CALL debug_message( debug_string, 'end' )
    ENDIF

 END SUBROUTINE usm_surface_energy_balance


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Swapping of time levels for t_surf and t_wall called out from subroutine swap_timelevel
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE usm_swap_timelevel( mod_count )

    IMPLICIT NONE

    INTEGER(iwp), INTENT(IN)  ::  mod_count  !<


    SELECT CASE ( mod_count )

       CASE ( 0 )
          t_surf_wall    => t_surf_wall_1;   t_surf_wall_p    => t_surf_wall_2
          t_wall         => t_wall_1;        t_wall_p         => t_wall_2
          t_surf_window  => t_surf_window_1; t_surf_window_p  => t_surf_window_2
          t_window       => t_window_1;      t_window_p       => t_window_2
          t_surf_green   => t_surf_green_1;  t_surf_green_p   => t_surf_green_2
          t_green        => t_green_1;       t_green_p        => t_green_2
       CASE ( 1 )
          t_surf_wall    => t_surf_wall_2;   t_surf_wall_p    => t_surf_wall_1
          t_wall         => t_wall_2;        t_wall_p         => t_wall_1
          t_surf_window  => t_surf_window_2; t_surf_window_p  => t_surf_window_1
          t_window       => t_window_2;      t_window_p       => t_window_1
          t_surf_green   => t_surf_green_2;  t_surf_green_p   => t_surf_green_1
          t_green        => t_green_2;       t_green_p        => t_green_1

    END SELECT

 END SUBROUTINE usm_swap_timelevel


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Calculate maximum allowed timestep at USM surfaces.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE usm_timestep
!
!-- Consider a pre-factor (1/8) for the diffusion criterion.
    IF ( ALLOCATED( surf_usm%dt_max ) )                                                            &
       dt_usm = MINVAL( surf_usm%dt_max ) * 0.125_wp
#if defined( __parallel )
    IF ( collective_wait )  CALL MPI_BARRIER( comm2d, ierr )
    CALL MPI_ALLREDUCE( MPI_IN_PLACE, dt_usm, 1, MPI_REAL, MPI_MIN, comm2d, ierr )
#endif

 END SUBROUTINE usm_timestep


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Sampling of USM variables along customized measurement coordinates.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE usm_vm_sampling( variable, var_atmos, i_atmos, j_atmos, k_atmos, ns_atmos,             &
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

    REAL(wp), DIMENSION(1:ns_soil) ::  var_soil  !< array to store soil variables


    SELECT CASE ( TRIM( variable ) )
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
                DO  mm = surf_usm%start_index(j,i), surf_usm%end_index(j,i)
                   var_soil(m) = MERGE( t_wall%val(k,mm), var_soil(m), surf_usm%upward(mm) )
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
       i_atmos = i_atmos
       j_atmos = j_atmos
       k_atmos = k_atmos
       var_atmos = var_atmos
    ENDIF

 END SUBROUTINE usm_vm_sampling


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Subroutine writes t_surf and t_wall data into restart files
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE usm_wrd_local

    IMPLICIT NONE

    INTEGER(idp), DIMENSION(nys:nyn,nxl:nxr) ::  global_end_index    !< end index for surface data (MPI-IO)
    INTEGER(idp), DIMENSION(nys:nyn,nxl:nxr) ::  global_start_index  !< start index for surface data (MPI-IO)

    LOGICAL  ::  surface_data_to_write  !< switch for MPI-I/O if PE has surface data to write


    IF ( TRIM( restart_data_format_output ) == 'fortran_binary' )  THEN

       CALL wrd_write_string( 'ns_on_file_usm' )
       WRITE ( 14 )  surf_usm%ns

       CALL wrd_write_string( 'usm_start_index' )
       WRITE ( 14 )  surf_usm%start_index

       CALL wrd_write_string( 'usm_end_index' )
       WRITE ( 14 )  surf_usm%end_index

       CALL wrd_write_string( 't_surf_wall' )
       WRITE ( 14 )  t_surf_wall%val

       CALL wrd_write_string( 't_surf_window' )
       WRITE ( 14 )  t_surf_window%val

       CALL wrd_write_string( 't_surf_green' )
       WRITE ( 14 )  t_surf_green%val

       CALL wrd_write_string( 'm_liq_usm' )
       WRITE ( 14 )  m_liq_usm%val

       CALL wrd_write_string( 'swc' )
       WRITE ( 14 )  swc%val

       CALL wrd_write_string( 't_wall' )
       WRITE ( 14 )  t_wall%val

       CALL wrd_write_string( 't_window' )
       WRITE ( 14 )  t_window%val

       CALL wrd_write_string( 't_green' )
       WRITE ( 14 )  t_green%val

    ELSEIF ( restart_data_format_output(1:3) == 'mpi' )  THEN
!
!--    There is no information about the PE-grid necessary because the restart files consists of the
!--    whole domain. Therefore, ns_on_file_usm are not used with MPI-IO.
       CALL rd_mpi_io_surface_filetypes( surf_usm%start_index, surf_usm%end_index,                 &
                                         surface_data_to_write, global_start_index,                &
                                         global_end_index )

       CALL wrd_mpi_io( 'usm_global_start', global_start_index )
       CALL wrd_mpi_io( 'usm_global_end', global_end_index )

       IF ( surface_data_to_write )  THEN
          CALL wrd_mpi_io_surface( 't_surf_wall',  t_surf_wall%val )
          CALL wrd_mpi_io_surface( 't_surf_window', t_surf_window%val )
          CALL wrd_mpi_io_surface( 't_surf_green', t_surf_green%val )

          CALL wrd_mpi_io_surface( 'm_liq_usm', m_liq_usm%val )
       ENDIF

       CALL rd_mpi_io_surface_filetypes( surf_usm%start_index, surf_usm%end_index,                 &
                                         surface_data_to_write, global_start_index,                &
                                         global_end_index )

       CALL wrd_mpi_io( 'usm_global_start_2', global_start_index )
       CALL wrd_mpi_io( 'usm_global_end_2', global_end_index )

       IF ( surface_data_to_write )  THEN
          CALL wrd_mpi_io_surface( 'swc', swc%val )
          CALL wrd_mpi_io_surface( 't_wall', t_wall%val )
          CALL wrd_mpi_io_surface( 't_window', t_window%val )
          CALL wrd_mpi_io_surface( 't_green', t_green%val )
       ENDIF

    ENDIF

 END SUBROUTINE usm_wrd_local


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Define building properties
!> Parameters 12, 13, 119 - 135 exclusive used in indoor_model_mod.f90
!> Parameters 0-11, 14-118, 136 - 149 exclusive used in urban_surface_mod.f90
!> Parameters 31, 44 used in indoor_model_mod.f90 and urban_surface_mod.f90
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE usm_define_pars

!
!-- Define the building_pars
    building_pars(:,1) = (/                                                                        &
       0.40_wp,        &  !< parameter 0   - [-] wall fraction above ground floor level
       0.60_wp,        &  !< parameter 1   - [-] window fraction above ground floor level
       0.0_wp,         &  !< parameter 2   - [-] green fraction above ground floor level
       0.0_wp,         &  !< parameter 3   - [-] green fraction roof above ground floor level
       2.0_wp,         &  !< parameter 4   - [m2/m2] LAI (Leaf Area Index) roof
       2.0_wp,         &  !< parameter 5   - [m2/m2] LAI (Leaf Area Index) on wall above ground floor level
       1520000.0_wp,   &  !< parameter 6   - [J/(m3*K)] heat capacity 1st wall layer (outside) above ground floor level
       2112000.0_wp,   &  !< parameter 7   - [J/(m3*K)] heat capacity 2nd wall layer above ground floor level
       2112000.0_wp,   &  !< parameter 8   - [J/(m3*K)] heat capacity 3rd wall layer above ground floor level
       0.93_wp,        &  !< parameter 9   - [W/(m*K)] thermal conductivity 1st wall layer (outside) above ground floor level
       2.10_wp,        &  !< parameter 10  - [W/(m*K)] thermal conductivity 2nd wall layer above ground floor level
       2.10_wp,        &  !< parameter 11  - [W/(m*K)] thermal conductivity 3rd wall layer above ground floor level
       301.15_wp,      &  !< parameter 12  - [K] indoor target summer temperature
       289.15_wp,      &  !< parameter 13  - [K] indoor target winter temperature
       0.93_wp,        &  !< parameter 14  - [-] wall emissivity above ground floor level
       0.86_wp,        &  !< parameter 15  - [-] green emissivity above ground floor level
       0.87_wp,        &  !< parameter 16  - [-] window emissivity above ground floor level
       0.65_wp,         &  !< parameter 17  - [-] window transmissivity (not visual transmissivity) above ground floor level
       0.001_wp,       &  !< parameter 18  - [m] z0 roughness above ground floor level
       0.0001_wp,      &  !< parameter 19  - [m] z0h/z0g roughness heat/humidity above ground floor level
       5.0_wp,         &  !< parameter 20  - [m] ground floor level height
       0.40_wp,        &  !< parameter 21  - [-] wall fraction ground floor level
       0.60_wp,        &  !< parameter 22  - [-] window fraction ground floor level
       0.0_wp,         &  !< parameter 23  - [-] green fraction ground floor level
       0.0_wp,         &  !< parameter 24  - [-] green fraction roof ground floor level
       2.0_wp,         &  !< parameter 25  - [m2/m2] LAI (Leaf Area Index) on wall ground floor level
       1520000.0_wp,   &  !< parameter 26  - [J/(m3*K)] heat capacity 1st wall layer (outside) ground floor level
       2112000.0_wp,   &  !< parameter 27  - [J/(m3*K)] heat capacity 2nd wall layer ground floor level
       2112000.0_wp,   &  !< parameter 28  - [J/(m3*K)] heat capacity 3rd wall layer ground floor level
       0.93_wp,        &  !< parameter 29  - [W/(m*K)] thermal conductivity 1st wall layer (outside) ground floor level
       2.10_wp,        &  !< parameter 30  - [W/(m*K)] thermal conductivity 2nd wall layer ground floor level
       2.10_wp,        &  !< parameter 31  - [W/(m*K)] thermal conductivity 3rd wall layer ground floor level
       0.93_wp,        &  !< parameter 32  - [-] wall emissivity ground floor level
       0.87_wp,        &  !< parameter 33  - [-] window emissivity ground floor level
       0.86_wp,        &  !< parameter 34  - [-] green emissivity ground floor level
       0.65_wp,         &  !< parameter 35  - [-] window transmissivity (not visual transmissivity) ground floor level
       0.001_wp,       &  !< parameter 36  - [m] z0 roughness ground floor level
       0.0001_wp,      &  !< parameter 37  - [m] z0h/z0q roughness heat/humidity
       36.0_wp,        &  !< parameter 38  - [-] wall albedo_type above ground floor level  (albedo_type specified in radiation model)
       5.0_wp,         &  !< parameter 39  - [-] green albedo_type above ground floor level  (albedo_type specified in radiation model)
       37.0_wp,        &  !< parameter 40  - [-] window albedo_type above ground floor level  (albedo_type specified in radiation model)
       0.02_wp,        &  !< parameter 41  - [m] 1st cumulative wall layer thickness above ground floor level
       0.17_wp,         &  !< parameter 42  - [m] 2nd cumulative wall layer thickness above ground floor level
       0.32_wp,        &  !< parameter 43  - [m] 3rd cumulative wall layer thickness above ground floor level
       0.34_wp,         &  !< parameter 44  - [m] 4th cumulative wall layer thickness above ground floor level
       20000.0_wp,     &  !< parameter 45  - [J/(m2*K)] heat capacity wall surface (1 cm air)
       23.0_wp,        &  !< parameter 46  - [W/(m2*K)] thermal conductivity of wall surface (1 cm air)
       20000.0_wp,     &  !< parameter 47  - [J/(m2*K)] heat capacity of window surface (1 cm air)
       20000.0_wp,     &  !< parameter 48  - [J/(m2*K)] heat capacity of green surface
       23.0_wp,        &  !< parameter 49  - [W/(m2*K)] thermal conductivity of window surface (1 cm air)
       10.0_wp,        &  !< parameter 50  - [W/(m2*K)] thermal conductivty of green surface
       1.0_wp,         &  !< parameter 51  - [-] wall fraction ground plate
       0.10_wp,        &  !< parameter 52  - [m] 1st cumulative wall layer thickness ground plate
       0.20_wp,        &  !< parameter 53  - [m] 2nd cumulative wall layer thickness ground plate
       0.26_wp,        &  !< parameter 54  - [m] 3rd cumulative wall layer thickness ground plate
       0.30_wp,        &  !< parameter 55  - [m] 4th cumulative wall layer thickness ground plate
       2112000.0_wp,   &  !< parameter 56  - [J/(m3*K)] heat capacity 1st wall layer (outside) ground plate
       2112000.0_wp,   &  !< parameter 57  - [J/(m3*K)] heat capacity 2nd wall layer ground plate
       2112000.0_wp,   &  !< parameter 58  - [J/(m3*K)] heat capacity 3rd wall layer ground plate
       2.10_wp,        &  !< parameter 59  - [W/(m*K)] thermal conductivity 1st wall layer (oustide) ground plate
       2.10_wp,        &  !< parameter 60  - [W/(m*K)] thermal conductivity 2nd wall layer ground plate
       2.1_wp,         &  !< parameter 61  - [W/(m*K)] thermal conductivity 3rd wall layer ground plate
       0.02_wp,        &  !< parameter 62  - [m] 1st cumulative wall layer thickness ground floor level
       0.17_wp,         &  !< parameter 63  - [m] 2nd cumulative wall layer thickness ground floor level
       0.32_wp,        &  !< parameter 64  - [m] 3rd cumulative wall layer thickness ground floor level
       0.34_wp,         &  !< parameter 65  - [m] 4th cumulative wall layer thickness ground floor level
       36.0_wp,        &  !< parameter 66  - [-] wall albedo_type ground floor level (albedo_type specified in radiation model)
       0.02_wp,        &  !< parameter 67  - [m] 1st cumulative window layer thickness ground floor level
       0.04_wp,        &  !< parameter 68  - [m] 2nd cumulative window layer thickness ground floor level
       0.06_wp,        &  !< parameter 69  - [m] 3rd cumulative window layer thickness ground floor level
       0.08_wp,        &  !< parameter 70  - [m] 4th cumulative window layer thickness ground floor level
       1736000.0_wp,   &  !< parameter 71  - [J/(m3*K)] heat capacity 1st window layer (outside) ground floor level
       1736000.0_wp,   &  !< parameter 72  - [J/(m3*K)] heat capacity 2nd window layer ground floor level
       1736000.0_wp,   &  !< parameter 73  - [J/(m3*K)] heat capacity 3rd window layer ground floor level
       0.19_wp,        &  !< parameter 74  - [W/(m*K)] thermal conductivity 1st window layer (outside) ground floor level
       0.19_wp,        &  !< parameter 75  - [W/(m*K)] thermal conductivity 2nd window layer ground floor level
       0.19_wp,        &  !< parameter 76  - [W/(m*K)] thermal conductivity 3rd window layer ground floor level
       37.0_wp,        &  !< parameter 77  - [-] window albedo_type ground floor level (albedo_type specified in radiation model)
       5.0_wp,         &  !< parameter 78  - [-] green albedo_type ground floor level (albedo_type specified in radiation model)
       0.02_wp,        &  !< parameter 79  - [m] 1st cumulative window layer thickness above ground floor level
       0.04_wp,        &  !< parameter 80  - [m] 2nd thickness window layer above ground floor level
       0.06_wp,        &  !< parameter 81  - [m] 3rd cumulative window layer thickness above ground floor level
       0.08_wp,        &  !< parameter 82  - [m] 4th cumulative window layer thickness above ground floor level
       1736000.0_wp,   &  !< parameter 83  - [J/(m3*K)] heat capacity 1st window layer (outside) above ground floor level
       1736000.0_wp,   &  !< parameter 84  - [J/(m3*K)] heat capacity 2nd window layer above ground floor level
       1736000.0_wp,   &  !< parameter 85  - [J/(m3*K)] heat capacity 3rd window layer above ground floor level
       0.19_wp,        &  !< parameter 86  - [W/(m*K)] thermal conductivity 1st window layer (outside) above ground floor level
       0.19_wp,        &  !< parameter 86  - [W/(m*K)] thermal conductivity 2nd window layer above ground floor level
       0.19_wp,        &  !< parameter 87  - [W/(m*K)] thermal conductivity 3rd window layer above ground floor level
       1.0_wp,         &  !< parameter 89  - [-] wall fraction roof
       0.02_wp,        &  !< parameter 90  - [m] 1st cumulative wall layer thickness roof
       0.17_wp,        &  !< parameter 91  - [m] 2nd cumulative wall layer thickness roof
       0.32_wp,        &  !< parameter 92  - [m] 3rd cumulative wall layer thickness roof
       0.34_wp,         &  !< parameter 93  - [m] 4th cumulative wall layer thickness roof
       1520000.0_wp,   &  !< parameter 94  - [J/(m3*K)] heat capacity 1st wall layer (outside) roof
       2112000.0_wp,    &  !< parameter 95  - [J/(m3*K)] heat capacity 2nd wall layer roof
       2112000.0_wp,    &  !< parameter 96  - [J/(m3*K)] heat capacity 3rd wall layer roof
       0.93_wp,        &  !< parameter 97  - [W/(m*K)] thermal conductivity 1st wall layer (outside) roof
       2.10_wp,        &  !< parameter 98  - [W/(m*K)] thermal conductivity 2nd wall layer roof
       2.10_wp,        &  !< parameter 99  - [W/(m*K)] thermal conductivity 3rd wall layer roof
       0.93_wp,        &  !< parameter 100 - [-] wall emissivity roof
       42.0_wp,        &  !< parameter 101 - [-] wall albedo_type roof (albedo_type specified in radiation model)
       0.0_wp,         &  !< parameter 102 - [-] window fraction roof
       0.02_wp,        &  !< parameter 103 - [m] window 1st layer thickness roof
       0.04_wp,        &  !< parameter 104 - [m] window 2nd layer thickness roof
       0.06_wp,        &  !< parameter 105 - [m] window 3rd layer thickness roof
       0.08_wp,        &  !< parameter 106 - [m] window 4th layer thickness roof
       1736000.0_wp,   &  !< parameter 107 - [J/(m3*K)] heat capacity 1st window layer (outside) roof
       1736000.0_wp,   &  !< parameter 108 - [J/(m3*K)] heat capacity 2nd window layer roof
       1736000.0_wp,   &  !< parameter 109 - [J/(m3*K)] heat capacity 3rd window layer roof
       0.19_wp,        &  !< parameter 110 - [W/(m*K)] thermal conductivity 1st window layer (outside) roof
       0.19_wp,        &  !< parameter 111 - [W/(m*K)] thermal conductivity 2nd window layer roof
       0.19_wp,        &  !< parameter 112 - [W/(m*K)] thermal conductivity 3rd window layer roof
       0.87_wp,        &  !< parameter 113 - [-] window emissivity roof
       0.65_wp,         &  !< parameter 114 - [-] window transmissivity (not visual transmissivity) roof
       37.0_wp,        &  !< parameter 115 - [-] window albedo_type roof (albedo_type specified in radiation model)
       0.86_wp,        &  !< parameter 116 - [-] green emissivity roof
       5.0_wp,         &  !< parameter 117 - [-] green albedo_type roof (albedo_type specified in radiation model)
       0.0_wp,         &  !< parameter 118 - [-] green type roof
       0.75_wp,        &  !< parameter 119 - [-] shading factor
       0.70_wp,         &  !< parameter 120 - [-] g-value windows
       1.7_wp,         &  !< parameter 121 - [W/(m2*K)] u-value windows
       0.5_wp,         &  !< parameter 122 - [1/h] basic airflow without occupancy of the room for - summer 0.5_wp, winter 0.5
       2.0_wp,         &  !< parameter 123 - [1/h] additional airflow dependent on occupancy of the room for - summer 1.5_wp, winter 0.0
       0.6_wp,         &  !< parameter 124 - [-] heat recovery efficiency
       2.5_wp,         &  !< parameter 125 - [m2/m2] dynamic parameter specific effective surface
       165000.0_wp,    &  !< parameter 126 - [J/(m2*K)] dynamic parameter innner heat storage
       4.5_wp,         &  !< parameter 127 - [m2/m2] ratio internal surface/floor area
       0.0_wp,       &  !< parameter 128 - [W] maximal heating capacity
       -80.0_wp,         &  !< parameter 129 - [W] maximal cooling capacity
       5.0_wp,         &  !< parameter 130 - [W/m2] additional internal heat gains dependent on occupancy of the room
       2.1_wp,         &  !< parameter 131 - [W/m2] basic internal heat gains without occupancy of the room
       5.0_wp,         &  !< parameter 132 - [m] storey height
       0.2_wp,         &  !< parameter 133 - [m] ceiling construction height
       0.0_wp,         &  !< parameter 134 - [-] anthropogenic heat output for heating
       1.0_wp,       &  !< parameter 135 - [-] anthropogenic heat output for cooling
       1526000.0_wp,   &  !< parameter 136 - [J/(m3*K)] heat capacity 4th wall layer (inside) above ground floor level
       0.93_wp,         &  !< parameter 137 - [W/(m*K)] thermal conductivity 4th wall layer (inside) above ground floor level
       1526000.0_wp,   &  !< parameter 138 - [J/(m3*K)] capacity 4th wall layer (inside) ground floor level
       0.93_wp,         &  !< parameter 139 - [W/(m*K)] thermal conductivity 4th wall layer (inside) ground floor level
       709650.0_wp,    &  !< parameter 140 - [J/(m3*K)] heat capacity 4th wall layer (inside) ground plate
       0.12_wp,        &  !< parameter 141 - [W/(m*K)] thermal conductivity 4th wall layer (inside) ground plate
       1736000.0_wp,   &  !< parameter 142 - [J/(m3*K)] heat capacity 4th window layer (inside) ground floor level
       0.19_wp,        &  !< parameter 143 - [W/(m*K)] thermal conductivity 4th window layer (inside) ground floor level
       1736000.0_wp,   &  !< parameter 144 - [J/(m3*K)] heat capacity 4th layer (inside) above ground floor level
       0.45_wp,        &  !< parameter 145 - [W/(m*K)] thermal conductivity 4th window layer (inside) above ground floor level
       1526000.0_wp,   &  !< parameter 146 - [J/(m3*K)] heat capacity 4th wall layer (inside) roof
       0.93_wp,         &  !< parameter 147 - [W/(m*K)] thermal conductivity 4th wall layer (inside) roof
       1736000.0_wp,   &  !< parameter 148 - [J/(m3*K)] heat capacity 4th window layer (inside) roof
       0.19_wp         &  !< parameter 149 - [W/(m*K)] thermal conductivity 4th window layer (inside) roof
                        /)

    building_pars(:,2) = (/                                                                        &
       0.40_wp,        &  !< parameter 0   - [-] wall fraction above ground floor level
       0.60_wp,        &  !< parameter 1   - [-] window fraction above ground floor level
       0.0_wp,         &  !< parameter 2   - [-] green fraction above ground floor level
       0.0_wp,         &  !< parameter 3   - [-] green fraction roof above ground floor level
       2.0_wp,         &  !< parameter 4   - [m2/m2] LAI (Leaf Area Index) roof
       2.0_wp,         &  !< parameter 5   - [m2/m2] LAI (Leaf Area Index) on wall above ground floor level
       1520000.0_wp,   &  !< parameter 6   - [J/(m3*K)] heat capacity 1st wall layer (outside) above ground floor level
       2112000.0_wp,   &  !< parameter 7   - [J/(m3*K)] heat capacity 2nd wall layer above ground floor level
       2112000.0_wp,   &  !< parameter 8   - [J/(m3*K)] heat capacity 3rd wall layer above ground floor level
       0.93_wp,        &  !< parameter 9   - [W/(m*K)] thermal conductivity 1st wall layer (outside) above ground floor level
       2.10_wp,        &  !< parameter 10  - [W/(m*K)] thermal conductivity 2nd wall layer above ground floor level
       2.10_wp,        &  !< parameter 11  - [W/(m*K)] thermal conductivity 3rd wall layer above ground floor level
       301.15_wp,      &  !< parameter 12  - [K] indoor target summer temperature
       289.15_wp,      &  !< parameter 13  - [K] indoor target winter temperature
       0.93_wp,        &  !< parameter 14  - [-] wall emissivity above ground floor level
       0.86_wp,        &  !< parameter 15  - [-] green emissivity above ground floor level
       0.87_wp,        &  !< parameter 16  - [-] window emissivity above ground floor level
       0.65_wp,         &  !< parameter 17  - [-] window transmissivity (not visual transmissivity) above ground floor level
       0.001_wp,       &  !< parameter 18  - [m] z0 roughness above ground floor level
       0.0001_wp,      &  !< parameter 19  - [m] z0h/z0g roughness heat/humidity above ground floor level
       5.0_wp,         &  !< parameter 20  - [m] ground floor level height
       0.40_wp,        &  !< parameter 21  - [-] wall fraction ground floor level
       0.60_wp,        &  !< parameter 22  - [-] window fraction ground floor level
       0.0_wp,         &  !< parameter 23  - [-] green fraction ground floor level
       0.0_wp,         &  !< parameter 24  - [-] green fraction roof ground floor level
       2.0_wp,         &  !< parameter 25  - [m2/m2] LAI (Leaf Area Index) on wall ground floor level
       1520000.0_wp,   &  !< parameter 26  - [J/(m3*K)] heat capacity 1st wall layer (outside) ground floor level
       2112000.0_wp,   &  !< parameter 27  - [J/(m3*K)] heat capacity 2nd wall layer ground floor level
       2112000.0_wp,   &  !< parameter 28  - [J/(m3*K)] heat capacity 3rd wall layer ground floor level
       0.93_wp,        &  !< parameter 29  - [W/(m*K)] thermal conductivity 1st wall layer (outside) ground floor level
       2.10_wp,        &  !< parameter 30  - [W/(m*K)] thermal conductivity 2nd wall layer ground floor level
       2.10_wp,        &  !< parameter 31  - [W/(m*K)] thermal conductivity 3rd wall layer ground floor level
       0.93_wp,        &  !< parameter 32  - [-] wall emissivity ground floor level
       0.87_wp,        &  !< parameter 33  - [-] window emissivity ground floor level
       0.86_wp,        &  !< parameter 34  - [-] green emissivity ground floor level
       0.65_wp,         &  !< parameter 35  - [-] window transmissivity (not visual transmissivity) ground floor level
       0.001_wp,       &  !< parameter 36  - [m] z0 roughness ground floor level
       0.0001_wp,      &  !< parameter 37  - [m] z0h/z0q roughness heat/humidity
       36.0_wp,        &  !< parameter 38  - [-] wall albedo_type above ground floor level  (albedo_type specified in radiation model)
       5.0_wp,         &  !< parameter 39  - [-] green albedo_type above ground floor level  (albedo_type specified in radiation model)
       37.0_wp,        &  !< parameter 40  - [-] window albedo_type above ground floor level  (albedo_type specified in radiation model)
       0.02_wp,        &  !< parameter 41  - [m] 1st cumulative wall layer thickness above ground floor level
       0.17_wp,         &  !< parameter 42  - [m] 2nd cumulative wall layer thickness above ground floor level
       0.32_wp,        &  !< parameter 43  - [m] 3rd cumulative wall layer thickness above ground floor level
       0.34_wp,         &  !< parameter 44  - [m] 4th cumulative wall layer thickness above ground floor level
       20000.0_wp,     &  !< parameter 45  - [J/(m2*K)] heat capacity wall surface (1 cm air)
       23.0_wp,        &  !< parameter 46  - [W/(m2*K)] thermal conductivity of wall surface (1 cm air)
       20000.0_wp,     &  !< parameter 47  - [J/(m2*K)] heat capacity of window surface (1 cm air)
       20000.0_wp,     &  !< parameter 48  - [J/(m2*K)] heat capacity of green surface
       23.0_wp,        &  !< parameter 49  - [W/(m2*K)] thermal conductivity of window surface (1 cm air)
       10.0_wp,        &  !< parameter 50  - [W/(m2*K)] thermal conductivty of green surface
       1.0_wp,         &  !< parameter 51  - [-] wall fraction ground plate
       0.10_wp,        &  !< parameter 52  - [m] 1st cumulative wall layer thickness ground plate
       0.20_wp,        &  !< parameter 53  - [m] 2nd cumulative wall layer thickness ground plate
       0.26_wp,        &  !< parameter 54  - [m] 3rd cumulative wall layer thickness ground plate
       0.30_wp,        &  !< parameter 55  - [m] 4th cumulative wall layer thickness ground plate
       2112000.0_wp,   &  !< parameter 56  - [J/(m3*K)] heat capacity 1st wall layer (outside) ground plate
       2112000.0_wp,   &  !< parameter 57  - [J/(m3*K)] heat capacity 2nd wall layer ground plate
       2112000.0_wp,   &  !< parameter 58  - [J/(m3*K)] heat capacity 3rd wall layer ground plate
       2.10_wp,        &  !< parameter 59  - [W/(m*K)] thermal conductivity 1st wall layer (oustide) ground plate
       2.10_wp,        &  !< parameter 60  - [W/(m*K)] thermal conductivity 2nd wall layer ground plate
       2.1_wp,         &  !< parameter 61  - [W/(m*K)] thermal conductivity 3rd wall layer ground plate
       0.02_wp,        &  !< parameter 62  - [m] 1st cumulative wall layer thickness ground floor level
       0.17_wp,         &  !< parameter 63  - [m] 2nd cumulative wall layer thickness ground floor level
       0.32_wp,        &  !< parameter 64  - [m] 3rd cumulative wall layer thickness ground floor level
       0.34_wp,         &  !< parameter 65  - [m] 4th cumulative wall layer thickness ground floor level
       36.0_wp,        &  !< parameter 66  - [-] wall albedo_type ground floor level (albedo_type specified in radiation model)
       0.02_wp,        &  !< parameter 67  - [m] 1st cumulative window layer thickness ground floor level
       0.04_wp,        &  !< parameter 68  - [m] 2nd cumulative window layer thickness ground floor level
       0.06_wp,        &  !< parameter 69  - [m] 3rd cumulative window layer thickness ground floor level
       0.08_wp,        &  !< parameter 70  - [m] 4th cumulative window layer thickness ground floor level
       1736000.0_wp,   &  !< parameter 71  - [J/(m3*K)] heat capacity 1st window layer (outside) ground floor level
       1736000.0_wp,   &  !< parameter 72  - [J/(m3*K)] heat capacity 2nd window layer ground floor level
       1736000.0_wp,   &  !< parameter 73  - [J/(m3*K)] heat capacity 3rd window layer ground floor level
       0.19_wp,        &  !< parameter 74  - [W/(m*K)] thermal conductivity 1st window layer (outside) ground floor level
       0.19_wp,        &  !< parameter 75  - [W/(m*K)] thermal conductivity 2nd window layer ground floor level
       0.19_wp,        &  !< parameter 76  - [W/(m*K)] thermal conductivity 3rd window layer ground floor level
       37.0_wp,        &  !< parameter 77  - [-] window albedo_type ground floor level (albedo_type specified in radiation model)
       5.0_wp,         &  !< parameter 78  - [-] green albedo_type ground floor level (albedo_type specified in radiation model)
       0.02_wp,        &  !< parameter 79  - [m] 1st cumulative window layer thickness above ground floor level
       0.04_wp,        &  !< parameter 80  - [m] 2nd thickness window layer above ground floor level
       0.06_wp,        &  !< parameter 81  - [m] 3rd cumulative window layer thickness above ground floor level
       0.08_wp,        &  !< parameter 82  - [m] 4th cumulative window layer thickness above ground floor level
       1736000.0_wp,   &  !< parameter 83  - [J/(m3*K)] heat capacity 1st window layer (outside) above ground floor level
       1736000.0_wp,   &  !< parameter 84  - [J/(m3*K)] heat capacity 2nd window layer above ground floor level
       1736000.0_wp,   &  !< parameter 85  - [J/(m3*K)] heat capacity 3rd window layer above ground floor level
       0.19_wp,        &  !< parameter 86  - [W/(m*K)] thermal conductivity 1st window layer (outside) above ground floor level
       0.19_wp,        &  !< parameter 86  - [W/(m*K)] thermal conductivity 2nd window layer above ground floor level
       0.19_wp,        &  !< parameter 87  - [W/(m*K)] thermal conductivity 3rd window layer above ground floor level
       1.0_wp,         &  !< parameter 89  - [-] wall fraction roof
       0.02_wp,        &  !< parameter 90  - [m] 1st cumulative wall layer thickness roof
       0.17_wp,        &  !< parameter 91  - [m] 2nd cumulative wall layer thickness roof
       0.32_wp,        &  !< parameter 92  - [m] 3rd cumulative wall layer thickness roof
       0.34_wp,         &  !< parameter 93  - [m] 4th cumulative wall layer thickness roof
       1520000.0_wp,   &  !< parameter 94  - [J/(m3*K)] heat capacity 1st wall layer (outside) roof
       2112000.0_wp,    &  !< parameter 95  - [J/(m3*K)] heat capacity 2nd wall layer roof
       2112000.0_wp,    &  !< parameter 96  - [J/(m3*K)] heat capacity 3rd wall layer roof
       0.93_wp,        &  !< parameter 97  - [W/(m*K)] thermal conductivity 1st wall layer (outside) roof
       2.10_wp,        &  !< parameter 98  - [W/(m*K)] thermal conductivity 2nd wall layer roof
       2.10_wp,        &  !< parameter 99  - [W/(m*K)] thermal conductivity 3rd wall layer roof
       0.93_wp,        &  !< parameter 100 - [-] wall emissivity roof
       42.0_wp,        &  !< parameter 101 - [-] wall albedo_type roof (albedo_type specified in radiation model)
       0.0_wp,         &  !< parameter 102 - [-] window fraction roof
       0.02_wp,        &  !< parameter 103 - [m] window 1st layer thickness roof
       0.04_wp,        &  !< parameter 104 - [m] window 2nd layer thickness roof
       0.06_wp,        &  !< parameter 105 - [m] window 3rd layer thickness roof
       0.08_wp,        &  !< parameter 106 - [m] window 4th layer thickness roof
       1736000.0_wp,   &  !< parameter 107 - [J/(m3*K)] heat capacity 1st window layer (outside) roof
       1736000.0_wp,   &  !< parameter 108 - [J/(m3*K)] heat capacity 2nd window layer roof
       1736000.0_wp,   &  !< parameter 109 - [J/(m3*K)] heat capacity 3rd window layer roof
       0.19_wp,        &  !< parameter 110 - [W/(m*K)] thermal conductivity 1st window layer (outside) roof
       0.19_wp,        &  !< parameter 111 - [W/(m*K)] thermal conductivity 2nd window layer roof
       0.19_wp,        &  !< parameter 112 - [W/(m*K)] thermal conductivity 3rd window layer roof
       0.87_wp,        &  !< parameter 113 - [-] window emissivity roof
       0.65_wp,         &  !< parameter 114 - [-] window transmissivity (not visual transmissivity) roof
       37.0_wp,        &  !< parameter 115 - [-] window albedo_type roof (albedo_type specified in radiation model)
       0.86_wp,        &  !< parameter 116 - [-] green emissivity roof
       5.0_wp,         &  !< parameter 117 - [-] green albedo_type roof (albedo_type specified in radiation model)
       0.0_wp,         &  !< parameter 118 - [-] green type roof
       0.75_wp,        &  !< parameter 119 - [-] shading factor
       0.70_wp,         &  !< parameter 120 - [-] g-value windows
       1.7_wp,         &  !< parameter 121 - [W/(m2*K)] u-value windows
       0.5_wp,         &  !< parameter 122 - [1/h] basic airflow without occupancy of the room for - summer 0.5_wp, winter 0.5
       2.0_wp,         &  !< parameter 123 - [1/h] additional airflow dependent on occupancy of the room for - summer 1.5_wp, winter 0.0
       0.6_wp,         &  !< parameter 124 - [-] heat recovery efficiency
       2.5_wp,         &  !< parameter 125 - [m2/m2] dynamic parameter specific effective surface
       165000.0_wp,    &  !< parameter 126 - [J/(m2*K)] dynamic parameter innner heat storage
       4.5_wp,         &  !< parameter 127 - [m2/m2] ratio internal surface/floor area
       0.0_wp,       &  !< parameter 128 - [W] maximal heating capacity
       -80.0_wp,         &  !< parameter 129 - [W] maximal cooling capacity
       5.0_wp,         &  !< parameter 130 - [W/m2] additional internal heat gains dependent on occupancy of the room
       2.1_wp,         &  !< parameter 131 - [W/m2] basic internal heat gains without occupancy of the room
       5.0_wp,         &  !< parameter 132 - [m] storey height
       0.2_wp,         &  !< parameter 133 - [m] ceiling construction height
       0.0_wp,         &  !< parameter 134 - [-] anthropogenic heat output for heating
       1.0_wp,       &  !< parameter 135 - [-] anthropogenic heat output for cooling
       1526000.0_wp,   &  !< parameter 136 - [J/(m3*K)] heat capacity 4th wall layer (inside) above ground floor level
       0.93_wp,         &  !< parameter 137 - [W/(m*K)] thermal conductivity 4th wall layer (inside) above ground floor level
       1526000.0_wp,   &  !< parameter 138 - [J/(m3*K)] capacity 4th wall layer (inside) ground floor level
       0.93_wp,         &  !< parameter 139 - [W/(m*K)] thermal conductivity 4th wall layer (inside) ground floor level
       709650.0_wp,    &  !< parameter 140 - [J/(m3*K)] heat capacity 4th wall layer (inside) ground plate
       0.12_wp,        &  !< parameter 141 - [W/(m*K)] thermal conductivity 4th wall layer (inside) ground plate
       1736000.0_wp,   &  !< parameter 142 - [J/(m3*K)] heat capacity 4th window layer (inside) ground floor level
       0.19_wp,        &  !< parameter 143 - [W/(m*K)] thermal conductivity 4th window layer (inside) ground floor level
       1736000.0_wp,   &  !< parameter 144 - [J/(m3*K)] heat capacity 4th layer (inside) above ground floor level
       0.45_wp,        &  !< parameter 145 - [W/(m*K)] thermal conductivity 4th window layer (inside) above ground floor level
       1526000.0_wp,   &  !< parameter 146 - [J/(m3*K)] heat capacity 4th wall layer (inside) roof
       0.93_wp,         &  !< parameter 147 - [W/(m*K)] thermal conductivity 4th wall layer (inside) roof
       1736000.0_wp,   &  !< parameter 148 - [J/(m3*K)] heat capacity 4th window layer (inside) roof
       0.19_wp         &  !< parameter 149 - [W/(m*K)] thermal conductivity 4th window layer (inside) roof
                        /)

    building_pars(:,3) = (/                                                                        &
       0.40_wp,        &  !< parameter 0   - [-] wall fraction above ground floor level
       0.60_wp,        &  !< parameter 1   - [-] window fraction above ground floor level
       0.0_wp,         &  !< parameter 2   - [-] green fraction above ground floor level
       0.0_wp,         &  !< parameter 3   - [-] green fraction roof above ground floor level
       2.0_wp,         &  !< parameter 4   - [m2/m2] LAI (Leaf Area Index) roof
       2.0_wp,         &  !< parameter 5   - [m2/m2] LAI (Leaf Area Index) on wall above ground floor level
       1520000.0_wp,   &  !< parameter 6   - [J/(m3*K)] heat capacity 1st wall layer (outside) above ground floor level
       2112000.0_wp,   &  !< parameter 7   - [J/(m3*K)] heat capacity 2nd wall layer above ground floor level
       2112000.0_wp,   &  !< parameter 8   - [J/(m3*K)] heat capacity 3rd wall layer above ground floor level
       0.93_wp,        &  !< parameter 9   - [W/(m*K)] thermal conductivity 1st wall layer (outside) above ground floor level
       2.10_wp,        &  !< parameter 10  - [W/(m*K)] thermal conductivity 2nd wall layer above ground floor level
       2.10_wp,        &  !< parameter 11  - [W/(m*K)] thermal conductivity 3rd wall layer above ground floor level
       301.15_wp,      &  !< parameter 12  - [K] indoor target summer temperature
       289.15_wp,      &  !< parameter 13  - [K] indoor target winter temperature
       0.93_wp,        &  !< parameter 14  - [-] wall emissivity above ground floor level
       0.86_wp,        &  !< parameter 15  - [-] green emissivity above ground floor level
       0.87_wp,        &  !< parameter 16  - [-] window emissivity above ground floor level
       0.65_wp,         &  !< parameter 17  - [-] window transmissivity (not visual transmissivity) above ground floor level
       0.001_wp,       &  !< parameter 18  - [m] z0 roughness above ground floor level
       0.0001_wp,      &  !< parameter 19  - [m] z0h/z0g roughness heat/humidity above ground floor level
       5.0_wp,         &  !< parameter 20  - [m] ground floor level height
       0.40_wp,        &  !< parameter 21  - [-] wall fraction ground floor level
       0.60_wp,        &  !< parameter 22  - [-] window fraction ground floor level
       0.0_wp,         &  !< parameter 23  - [-] green fraction ground floor level
       0.0_wp,         &  !< parameter 24  - [-] green fraction roof ground floor level
       2.0_wp,         &  !< parameter 25  - [m2/m2] LAI (Leaf Area Index) on wall ground floor level
       1520000.0_wp,   &  !< parameter 26  - [J/(m3*K)] heat capacity 1st wall layer (outside) ground floor level
       2112000.0_wp,   &  !< parameter 27  - [J/(m3*K)] heat capacity 2nd wall layer ground floor level
       2112000.0_wp,   &  !< parameter 28  - [J/(m3*K)] heat capacity 3rd wall layer ground floor level
       0.93_wp,        &  !< parameter 29  - [W/(m*K)] thermal conductivity 1st wall layer (outside) ground floor level
       2.10_wp,        &  !< parameter 30  - [W/(m*K)] thermal conductivity 2nd wall layer ground floor level
       2.10_wp,        &  !< parameter 31  - [W/(m*K)] thermal conductivity 3rd wall layer ground floor level
       0.93_wp,        &  !< parameter 32  - [-] wall emissivity ground floor level
       0.87_wp,        &  !< parameter 33  - [-] window emissivity ground floor level
       0.86_wp,        &  !< parameter 34  - [-] green emissivity ground floor level
       0.65_wp,         &  !< parameter 35  - [-] window transmissivity (not visual transmissivity) ground floor level
       0.001_wp,       &  !< parameter 36  - [m] z0 roughness ground floor level
       0.0001_wp,      &  !< parameter 37  - [m] z0h/z0q roughness heat/humidity
       36.0_wp,        &  !< parameter 38  - [-] wall albedo_type above ground floor level  (albedo_type specified in radiation model)
       5.0_wp,         &  !< parameter 39  - [-] green albedo_type above ground floor level  (albedo_type specified in radiation model)
       37.0_wp,        &  !< parameter 40  - [-] window albedo_type above ground floor level  (albedo_type specified in radiation model)
       0.02_wp,        &  !< parameter 41  - [m] 1st cumulative wall layer thickness above ground floor level
       0.17_wp,         &  !< parameter 42  - [m] 2nd cumulative wall layer thickness above ground floor level
       0.32_wp,        &  !< parameter 43  - [m] 3rd cumulative wall layer thickness above ground floor level
       0.34_wp,         &  !< parameter 44  - [m] 4th cumulative wall layer thickness above ground floor level
       20000.0_wp,     &  !< parameter 45  - [J/(m2*K)] heat capacity wall surface (1 cm air)
       23.0_wp,        &  !< parameter 46  - [W/(m2*K)] thermal conductivity of wall surface (1 cm air)
       20000.0_wp,     &  !< parameter 47  - [J/(m2*K)] heat capacity of window surface (1 cm air)
       20000.0_wp,     &  !< parameter 48  - [J/(m2*K)] heat capacity of green surface
       23.0_wp,        &  !< parameter 49  - [W/(m2*K)] thermal conductivity of window surface (1 cm air)
       10.0_wp,        &  !< parameter 50  - [W/(m2*K)] thermal conductivty of green surface
       1.0_wp,         &  !< parameter 51  - [-] wall fraction ground plate
       0.10_wp,        &  !< parameter 52  - [m] 1st cumulative wall layer thickness ground plate
       0.20_wp,        &  !< parameter 53  - [m] 2nd cumulative wall layer thickness ground plate
       0.26_wp,        &  !< parameter 54  - [m] 3rd cumulative wall layer thickness ground plate
       0.30_wp,        &  !< parameter 55  - [m] 4th cumulative wall layer thickness ground plate
       2112000.0_wp,   &  !< parameter 56  - [J/(m3*K)] heat capacity 1st wall layer (outside) ground plate
       2112000.0_wp,   &  !< parameter 57  - [J/(m3*K)] heat capacity 2nd wall layer ground plate
       2112000.0_wp,   &  !< parameter 58  - [J/(m3*K)] heat capacity 3rd wall layer ground plate
       2.10_wp,        &  !< parameter 59  - [W/(m*K)] thermal conductivity 1st wall layer (oustide) ground plate
       2.10_wp,        &  !< parameter 60  - [W/(m*K)] thermal conductivity 2nd wall layer ground plate
       2.1_wp,         &  !< parameter 61  - [W/(m*K)] thermal conductivity 3rd wall layer ground plate
       0.02_wp,        &  !< parameter 62  - [m] 1st cumulative wall layer thickness ground floor level
       0.17_wp,         &  !< parameter 63  - [m] 2nd cumulative wall layer thickness ground floor level
       0.32_wp,        &  !< parameter 64  - [m] 3rd cumulative wall layer thickness ground floor level
       0.34_wp,         &  !< parameter 65  - [m] 4th cumulative wall layer thickness ground floor level
       36.0_wp,        &  !< parameter 66  - [-] wall albedo_type ground floor level (albedo_type specified in radiation model)
       0.02_wp,        &  !< parameter 67  - [m] 1st cumulative window layer thickness ground floor level
       0.04_wp,        &  !< parameter 68  - [m] 2nd cumulative window layer thickness ground floor level
       0.06_wp,        &  !< parameter 69  - [m] 3rd cumulative window layer thickness ground floor level
       0.08_wp,        &  !< parameter 70  - [m] 4th cumulative window layer thickness ground floor level
       1736000.0_wp,   &  !< parameter 71  - [J/(m3*K)] heat capacity 1st window layer (outside) ground floor level
       1736000.0_wp,   &  !< parameter 72  - [J/(m3*K)] heat capacity 2nd window layer ground floor level
       1736000.0_wp,   &  !< parameter 73  - [J/(m3*K)] heat capacity 3rd window layer ground floor level
       0.19_wp,        &  !< parameter 74  - [W/(m*K)] thermal conductivity 1st window layer (outside) ground floor level
       0.19_wp,        &  !< parameter 75  - [W/(m*K)] thermal conductivity 2nd window layer ground floor level
       0.19_wp,        &  !< parameter 76  - [W/(m*K)] thermal conductivity 3rd window layer ground floor level
       37.0_wp,        &  !< parameter 77  - [-] window albedo_type ground floor level (albedo_type specified in radiation model)
       5.0_wp,         &  !< parameter 78  - [-] green albedo_type ground floor level (albedo_type specified in radiation model)
       0.02_wp,        &  !< parameter 79  - [m] 1st cumulative window layer thickness above ground floor level
       0.04_wp,        &  !< parameter 80  - [m] 2nd thickness window layer above ground floor level
       0.06_wp,        &  !< parameter 81  - [m] 3rd cumulative window layer thickness above ground floor level
       0.08_wp,        &  !< parameter 82  - [m] 4th cumulative window layer thickness above ground floor level
       1736000.0_wp,   &  !< parameter 83  - [J/(m3*K)] heat capacity 1st window layer (outside) above ground floor level
       1736000.0_wp,   &  !< parameter 84  - [J/(m3*K)] heat capacity 2nd window layer above ground floor level
       1736000.0_wp,   &  !< parameter 85  - [J/(m3*K)] heat capacity 3rd window layer above ground floor level
       0.19_wp,        &  !< parameter 86  - [W/(m*K)] thermal conductivity 1st window layer (outside) above ground floor level
       0.19_wp,        &  !< parameter 86  - [W/(m*K)] thermal conductivity 2nd window layer above ground floor level
       0.19_wp,        &  !< parameter 87  - [W/(m*K)] thermal conductivity 3rd window layer above ground floor level
       1.0_wp,         &  !< parameter 89  - [-] wall fraction roof
       0.02_wp,        &  !< parameter 90  - [m] 1st cumulative wall layer thickness roof
       0.17_wp,        &  !< parameter 91  - [m] 2nd cumulative wall layer thickness roof
       0.32_wp,        &  !< parameter 92  - [m] 3rd cumulative wall layer thickness roof
       0.34_wp,         &  !< parameter 93  - [m] 4th cumulative wall layer thickness roof
       1520000.0_wp,   &  !< parameter 94  - [J/(m3*K)] heat capacity 1st wall layer (outside) roof
       2112000.0_wp,    &  !< parameter 95  - [J/(m3*K)] heat capacity 2nd wall layer roof
       2112000.0_wp,    &  !< parameter 96  - [J/(m3*K)] heat capacity 3rd wall layer roof
       0.93_wp,        &  !< parameter 97  - [W/(m*K)] thermal conductivity 1st wall layer (outside) roof
       2.10_wp,        &  !< parameter 98  - [W/(m*K)] thermal conductivity 2nd wall layer roof
       2.10_wp,        &  !< parameter 99  - [W/(m*K)] thermal conductivity 3rd wall layer roof
       0.93_wp,        &  !< parameter 100 - [-] wall emissivity roof
       42.0_wp,        &  !< parameter 101 - [-] wall albedo_type roof (albedo_type specified in radiation model)
       0.0_wp,         &  !< parameter 102 - [-] window fraction roof
       0.02_wp,        &  !< parameter 103 - [m] window 1st layer thickness roof
       0.04_wp,        &  !< parameter 104 - [m] window 2nd layer thickness roof
       0.06_wp,        &  !< parameter 105 - [m] window 3rd layer thickness roof
       0.08_wp,        &  !< parameter 106 - [m] window 4th layer thickness roof
       1736000.0_wp,   &  !< parameter 107 - [J/(m3*K)] heat capacity 1st window layer (outside) roof
       1736000.0_wp,   &  !< parameter 108 - [J/(m3*K)] heat capacity 2nd window layer roof
       1736000.0_wp,   &  !< parameter 109 - [J/(m3*K)] heat capacity 3rd window layer roof
       0.19_wp,        &  !< parameter 110 - [W/(m*K)] thermal conductivity 1st window layer (outside) roof
       0.19_wp,        &  !< parameter 111 - [W/(m*K)] thermal conductivity 2nd window layer roof
       0.19_wp,        &  !< parameter 112 - [W/(m*K)] thermal conductivity 3rd window layer roof
       0.87_wp,        &  !< parameter 113 - [-] window emissivity roof
       0.65_wp,         &  !< parameter 114 - [-] window transmissivity (not visual transmissivity) roof
       37.0_wp,        &  !< parameter 115 - [-] window albedo_type roof (albedo_type specified in radiation model)
       0.86_wp,        &  !< parameter 116 - [-] green emissivity roof
       5.0_wp,         &  !< parameter 117 - [-] green albedo_type roof (albedo_type specified in radiation model)
       0.0_wp,         &  !< parameter 118 - [-] green type roof
       0.75_wp,        &  !< parameter 119 - [-] shading factor
       0.70_wp,         &  !< parameter 120 - [-] g-value windows
       1.7_wp,         &  !< parameter 121 - [W/(m2*K)] u-value windows
       0.5_wp,         &  !< parameter 122 - [1/h] basic airflow without occupancy of the room for - summer 0.5_wp, winter 0.5
       2.0_wp,         &  !< parameter 123 - [1/h] additional airflow dependent on occupancy of the room for - summer 1.5_wp, winter 0.0
       0.6_wp,         &  !< parameter 124 - [-] heat recovery efficiency
       2.5_wp,         &  !< parameter 125 - [m2/m2] dynamic parameter specific effective surface
       165000.0_wp,    &  !< parameter 126 - [J/(m2*K)] dynamic parameter innner heat storage
       4.5_wp,         &  !< parameter 127 - [m2/m2] ratio internal surface/floor area
       0.0_wp,       &  !< parameter 128 - [W] maximal heating capacity
       -80.0_wp,         &  !< parameter 129 - [W] maximal cooling capacity
       5.0_wp,         &  !< parameter 130 - [W/m2] additional internal heat gains dependent on occupancy of the room
       2.1_wp,         &  !< parameter 131 - [W/m2] basic internal heat gains without occupancy of the room
       5.0_wp,         &  !< parameter 132 - [m] storey height
       0.2_wp,         &  !< parameter 133 - [m] ceiling construction height
       0.0_wp,         &  !< parameter 134 - [-] anthropogenic heat output for heating
       1.0_wp,       &  !< parameter 135 - [-] anthropogenic heat output for cooling
       1526000.0_wp,   &  !< parameter 136 - [J/(m3*K)] heat capacity 4th wall layer (inside) above ground floor level
       0.93_wp,         &  !< parameter 137 - [W/(m*K)] thermal conductivity 4th wall layer (inside) above ground floor level
       1526000.0_wp,   &  !< parameter 138 - [J/(m3*K)] capacity 4th wall layer (inside) ground floor level
       0.93_wp,         &  !< parameter 139 - [W/(m*K)] thermal conductivity 4th wall layer (inside) ground floor level
       709650.0_wp,    &  !< parameter 140 - [J/(m3*K)] heat capacity 4th wall layer (inside) ground plate
       0.12_wp,        &  !< parameter 141 - [W/(m*K)] thermal conductivity 4th wall layer (inside) ground plate
       1736000.0_wp,   &  !< parameter 142 - [J/(m3*K)] heat capacity 4th window layer (inside) ground floor level
       0.19_wp,        &  !< parameter 143 - [W/(m*K)] thermal conductivity 4th window layer (inside) ground floor level
       1736000.0_wp,   &  !< parameter 144 - [J/(m3*K)] heat capacity 4th layer (inside) above ground floor level
       0.45_wp,        &  !< parameter 145 - [W/(m*K)] thermal conductivity 4th window layer (inside) above ground floor level
       1526000.0_wp,   &  !< parameter 146 - [J/(m3*K)] heat capacity 4th wall layer (inside) roof
       0.93_wp,         &  !< parameter 147 - [W/(m*K)] thermal conductivity 4th wall layer (inside) roof
       1736000.0_wp,   &  !< parameter 148 - [J/(m3*K)] heat capacity 4th window layer (inside) roof
       0.19_wp         &  !< parameter 149 - [W/(m*K)] thermal conductivity 4th window layer (inside) roof
                        /)

   building_pars(:,4) = (/                                                                        &
       0.40_wp,        &  !< parameter 0   - [-] wall fraction above ground floor level
       0.60_wp,        &  !< parameter 1   - [-] window fraction above ground floor level
       0.0_wp,         &  !< parameter 2   - [-] green fraction above ground floor level
       0.0_wp,         &  !< parameter 3   - [-] green fraction roof above ground floor level
       2.0_wp,         &  !< parameter 4   - [m2/m2] LAI (Leaf Area Index) roof
       2.0_wp,         &  !< parameter 5   - [m2/m2] LAI (Leaf Area Index) on wall above ground floor level
       1520000.0_wp,   &  !< parameter 6   - [J/(m3*K)] heat capacity 1st wall layer (outside) above ground floor level
       2112000.0_wp,   &  !< parameter 7   - [J/(m3*K)] heat capacity 2nd wall layer above ground floor level
       2112000.0_wp,   &  !< parameter 8   - [J/(m3*K)] heat capacity 3rd wall layer above ground floor level
       0.93_wp,        &  !< parameter 9   - [W/(m*K)] thermal conductivity 1st wall layer (outside) above ground floor level
       2.10_wp,        &  !< parameter 10  - [W/(m*K)] thermal conductivity 2nd wall layer above ground floor level
       2.10_wp,        &  !< parameter 11  - [W/(m*K)] thermal conductivity 3rd wall layer above ground floor level
       301.15_wp,      &  !< parameter 12  - [K] indoor target summer temperature
       289.15_wp,      &  !< parameter 13  - [K] indoor target winter temperature
       0.93_wp,        &  !< parameter 14  - [-] wall emissivity above ground floor level
       0.86_wp,        &  !< parameter 15  - [-] green emissivity above ground floor level
       0.87_wp,        &  !< parameter 16  - [-] window emissivity above ground floor level
       0.65_wp,         &  !< parameter 17  - [-] window transmissivity (not visual transmissivity) above ground floor level
       0.001_wp,       &  !< parameter 18  - [m] z0 roughness above ground floor level
       0.0001_wp,      &  !< parameter 19  - [m] z0h/z0g roughness heat/humidity above ground floor level
       5.0_wp,         &  !< parameter 20  - [m] ground floor level height
       0.40_wp,        &  !< parameter 21  - [-] wall fraction ground floor level
       0.60_wp,        &  !< parameter 22  - [-] window fraction ground floor level
       0.0_wp,         &  !< parameter 23  - [-] green fraction ground floor level
       0.0_wp,         &  !< parameter 24  - [-] green fraction roof ground floor level
       2.0_wp,         &  !< parameter 25  - [m2/m2] LAI (Leaf Area Index) on wall ground floor level
       1520000.0_wp,   &  !< parameter 26  - [J/(m3*K)] heat capacity 1st wall layer (outside) ground floor level
       2112000.0_wp,   &  !< parameter 27  - [J/(m3*K)] heat capacity 2nd wall layer ground floor level
       2112000.0_wp,   &  !< parameter 28  - [J/(m3*K)] heat capacity 3rd wall layer ground floor level
       0.93_wp,        &  !< parameter 29  - [W/(m*K)] thermal conductivity 1st wall layer (outside) ground floor level
       2.10_wp,        &  !< parameter 30  - [W/(m*K)] thermal conductivity 2nd wall layer ground floor level
       2.10_wp,        &  !< parameter 31  - [W/(m*K)] thermal conductivity 3rd wall layer ground floor level
       0.93_wp,        &  !< parameter 32  - [-] wall emissivity ground floor level
       0.87_wp,        &  !< parameter 33  - [-] window emissivity ground floor level
       0.86_wp,        &  !< parameter 34  - [-] green emissivity ground floor level
       0.65_wp,         &  !< parameter 35  - [-] window transmissivity (not visual transmissivity) ground floor level
       0.001_wp,       &  !< parameter 36  - [m] z0 roughness ground floor level
       0.0001_wp,      &  !< parameter 37  - [m] z0h/z0q roughness heat/humidity
       36.0_wp,        &  !< parameter 38  - [-] wall albedo_type above ground floor level  (albedo_type specified in radiation model)
       5.0_wp,         &  !< parameter 39  - [-] green albedo_type above ground floor level  (albedo_type specified in radiation model)
       37.0_wp,        &  !< parameter 40  - [-] window albedo_type above ground floor level  (albedo_type specified in radiation model)
       0.02_wp,        &  !< parameter 41  - [m] 1st cumulative wall layer thickness above ground floor level
       0.17_wp,         &  !< parameter 42  - [m] 2nd cumulative wall layer thickness above ground floor level
       0.32_wp,        &  !< parameter 43  - [m] 3rd cumulative wall layer thickness above ground floor level
       0.34_wp,         &  !< parameter 44  - [m] 4th cumulative wall layer thickness above ground floor level
       20000.0_wp,     &  !< parameter 45  - [J/(m2*K)] heat capacity wall surface (1 cm air)
       23.0_wp,        &  !< parameter 46  - [W/(m2*K)] thermal conductivity of wall surface (1 cm air)
       20000.0_wp,     &  !< parameter 47  - [J/(m2*K)] heat capacity of window surface (1 cm air)
       20000.0_wp,     &  !< parameter 48  - [J/(m2*K)] heat capacity of green surface
       23.0_wp,        &  !< parameter 49  - [W/(m2*K)] thermal conductivity of window surface (1 cm air)
       10.0_wp,        &  !< parameter 50  - [W/(m2*K)] thermal conductivty of green surface
       1.0_wp,         &  !< parameter 51  - [-] wall fraction ground plate
       0.10_wp,        &  !< parameter 52  - [m] 1st cumulative wall layer thickness ground plate
       0.20_wp,        &  !< parameter 53  - [m] 2nd cumulative wall layer thickness ground plate
       0.26_wp,        &  !< parameter 54  - [m] 3rd cumulative wall layer thickness ground plate
       0.30_wp,        &  !< parameter 55  - [m] 4th cumulative wall layer thickness ground plate
       2112000.0_wp,   &  !< parameter 56  - [J/(m3*K)] heat capacity 1st wall layer (outside) ground plate
       2112000.0_wp,   &  !< parameter 57  - [J/(m3*K)] heat capacity 2nd wall layer ground plate
       2112000.0_wp,   &  !< parameter 58  - [J/(m3*K)] heat capacity 3rd wall layer ground plate
       2.10_wp,        &  !< parameter 59  - [W/(m*K)] thermal conductivity 1st wall layer (oustide) ground plate
       2.10_wp,        &  !< parameter 60  - [W/(m*K)] thermal conductivity 2nd wall layer ground plate
       2.1_wp,         &  !< parameter 61  - [W/(m*K)] thermal conductivity 3rd wall layer ground plate
       0.02_wp,        &  !< parameter 62  - [m] 1st cumulative wall layer thickness ground floor level
       0.17_wp,         &  !< parameter 63  - [m] 2nd cumulative wall layer thickness ground floor level
       0.32_wp,        &  !< parameter 64  - [m] 3rd cumulative wall layer thickness ground floor level
       0.34_wp,         &  !< parameter 65  - [m] 4th cumulative wall layer thickness ground floor level
       36.0_wp,        &  !< parameter 66  - [-] wall albedo_type ground floor level (albedo_type specified in radiation model)
       0.02_wp,        &  !< parameter 67  - [m] 1st cumulative window layer thickness ground floor level
       0.04_wp,        &  !< parameter 68  - [m] 2nd cumulative window layer thickness ground floor level
       0.06_wp,        &  !< parameter 69  - [m] 3rd cumulative window layer thickness ground floor level
       0.08_wp,        &  !< parameter 70  - [m] 4th cumulative window layer thickness ground floor level
       1736000.0_wp,   &  !< parameter 71  - [J/(m3*K)] heat capacity 1st window layer (outside) ground floor level
       1736000.0_wp,   &  !< parameter 72  - [J/(m3*K)] heat capacity 2nd window layer ground floor level
       1736000.0_wp,   &  !< parameter 73  - [J/(m3*K)] heat capacity 3rd window layer ground floor level
       0.19_wp,        &  !< parameter 74  - [W/(m*K)] thermal conductivity 1st window layer (outside) ground floor level
       0.19_wp,        &  !< parameter 75  - [W/(m*K)] thermal conductivity 2nd window layer ground floor level
       0.19_wp,        &  !< parameter 76  - [W/(m*K)] thermal conductivity 3rd window layer ground floor level
       37.0_wp,        &  !< parameter 77  - [-] window albedo_type ground floor level (albedo_type specified in radiation model)
       5.0_wp,         &  !< parameter 78  - [-] green albedo_type ground floor level (albedo_type specified in radiation model)
       0.02_wp,        &  !< parameter 79  - [m] 1st cumulative window layer thickness above ground floor level
       0.04_wp,        &  !< parameter 80  - [m] 2nd thickness window layer above ground floor level
       0.06_wp,        &  !< parameter 81  - [m] 3rd cumulative window layer thickness above ground floor level
       0.08_wp,        &  !< parameter 82  - [m] 4th cumulative window layer thickness above ground floor level
       1736000.0_wp,   &  !< parameter 83  - [J/(m3*K)] heat capacity 1st window layer (outside) above ground floor level
       1736000.0_wp,   &  !< parameter 84  - [J/(m3*K)] heat capacity 2nd window layer above ground floor level
       1736000.0_wp,   &  !< parameter 85  - [J/(m3*K)] heat capacity 3rd window layer above ground floor level
       0.19_wp,        &  !< parameter 86  - [W/(m*K)] thermal conductivity 1st window layer (outside) above ground floor level
       0.19_wp,        &  !< parameter 86  - [W/(m*K)] thermal conductivity 2nd window layer above ground floor level
       0.19_wp,        &  !< parameter 87  - [W/(m*K)] thermal conductivity 3rd window layer above ground floor level
       1.0_wp,         &  !< parameter 89  - [-] wall fraction roof
       0.02_wp,        &  !< parameter 90  - [m] 1st cumulative wall layer thickness roof
       0.17_wp,        &  !< parameter 91  - [m] 2nd cumulative wall layer thickness roof
       0.32_wp,        &  !< parameter 92  - [m] 3rd cumulative wall layer thickness roof
       0.34_wp,         &  !< parameter 93  - [m] 4th cumulative wall layer thickness roof
       1520000.0_wp,   &  !< parameter 94  - [J/(m3*K)] heat capacity 1st wall layer (outside) roof
       2112000.0_wp,    &  !< parameter 95  - [J/(m3*K)] heat capacity 2nd wall layer roof
       2112000.0_wp,    &  !< parameter 96  - [J/(m3*K)] heat capacity 3rd wall layer roof
       0.93_wp,        &  !< parameter 97  - [W/(m*K)] thermal conductivity 1st wall layer (outside) roof
       2.10_wp,        &  !< parameter 98  - [W/(m*K)] thermal conductivity 2nd wall layer roof
       2.10_wp,        &  !< parameter 99  - [W/(m*K)] thermal conductivity 3rd wall layer roof
       0.93_wp,        &  !< parameter 100 - [-] wall emissivity roof
       42.0_wp,        &  !< parameter 101 - [-] wall albedo_type roof (albedo_type specified in radiation model)
       0.0_wp,         &  !< parameter 102 - [-] window fraction roof
       0.02_wp,        &  !< parameter 103 - [m] window 1st layer thickness roof
       0.04_wp,        &  !< parameter 104 - [m] window 2nd layer thickness roof
       0.06_wp,        &  !< parameter 105 - [m] window 3rd layer thickness roof
       0.08_wp,        &  !< parameter 106 - [m] window 4th layer thickness roof
       1736000.0_wp,   &  !< parameter 107 - [J/(m3*K)] heat capacity 1st window layer (outside) roof
       1736000.0_wp,   &  !< parameter 108 - [J/(m3*K)] heat capacity 2nd window layer roof
       1736000.0_wp,   &  !< parameter 109 - [J/(m3*K)] heat capacity 3rd window layer roof
       0.19_wp,        &  !< parameter 110 - [W/(m*K)] thermal conductivity 1st window layer (outside) roof
       0.19_wp,        &  !< parameter 111 - [W/(m*K)] thermal conductivity 2nd window layer roof
       0.19_wp,        &  !< parameter 112 - [W/(m*K)] thermal conductivity 3rd window layer roof
       0.87_wp,        &  !< parameter 113 - [-] window emissivity roof
       0.65_wp,         &  !< parameter 114 - [-] window transmissivity (not visual transmissivity) roof
       37.0_wp,        &  !< parameter 115 - [-] window albedo_type roof (albedo_type specified in radiation model)
       0.86_wp,        &  !< parameter 116 - [-] green emissivity roof
       5.0_wp,         &  !< parameter 117 - [-] green albedo_type roof (albedo_type specified in radiation model)
       0.0_wp,         &  !< parameter 118 - [-] green type roof
       0.75_wp,        &  !< parameter 119 - [-] shading factor
       0.70_wp,         &  !< parameter 120 - [-] g-value windows
       1.7_wp,         &  !< parameter 121 - [W/(m2*K)] u-value windows
       0.5_wp,         &  !< parameter 122 - [1/h] basic airflow without occupancy of the room for - summer 0.5_wp, winter 0.5
       2.0_wp,         &  !< parameter 123 - [1/h] additional airflow dependent on occupancy of the room for - summer 1.5_wp, winter 0.0
       0.6_wp,         &  !< parameter 124 - [-] heat recovery efficiency
       2.5_wp,         &  !< parameter 125 - [m2/m2] dynamic parameter specific effective surface
       165000.0_wp,    &  !< parameter 126 - [J/(m2*K)] dynamic parameter innner heat storage
       4.5_wp,         &  !< parameter 127 - [m2/m2] ratio internal surface/floor area
       0.0_wp,       &  !< parameter 128 - [W] maximal heating capacity
       -80.0_wp,         &  !< parameter 129 - [W] maximal cooling capacity
       5.0_wp,         &  !< parameter 130 - [W/m2] additional internal heat gains dependent on occupancy of the room
       2.1_wp,         &  !< parameter 131 - [W/m2] basic internal heat gains without occupancy of the room
       5.0_wp,         &  !< parameter 132 - [m] storey height
       0.2_wp,         &  !< parameter 133 - [m] ceiling construction height
       0.0_wp,         &  !< parameter 134 - [-] anthropogenic heat output for heating
       1.0_wp,       &  !< parameter 135 - [-] anthropogenic heat output for cooling
       1526000.0_wp,   &  !< parameter 136 - [J/(m3*K)] heat capacity 4th wall layer (inside) above ground floor level
       0.93_wp,         &  !< parameter 137 - [W/(m*K)] thermal conductivity 4th wall layer (inside) above ground floor level
       1526000.0_wp,   &  !< parameter 138 - [J/(m3*K)] capacity 4th wall layer (inside) ground floor level
       0.93_wp,         &  !< parameter 139 - [W/(m*K)] thermal conductivity 4th wall layer (inside) ground floor level
       709650.0_wp,    &  !< parameter 140 - [J/(m3*K)] heat capacity 4th wall layer (inside) ground plate
       0.12_wp,        &  !< parameter 141 - [W/(m*K)] thermal conductivity 4th wall layer (inside) ground plate
       1736000.0_wp,   &  !< parameter 142 - [J/(m3*K)] heat capacity 4th window layer (inside) ground floor level
       0.19_wp,        &  !< parameter 143 - [W/(m*K)] thermal conductivity 4th window layer (inside) ground floor level
       1736000.0_wp,   &  !< parameter 144 - [J/(m3*K)] heat capacity 4th layer (inside) above ground floor level
       0.45_wp,        &  !< parameter 145 - [W/(m*K)] thermal conductivity 4th window layer (inside) above ground floor level
       1526000.0_wp,   &  !< parameter 146 - [J/(m3*K)] heat capacity 4th wall layer (inside) roof
       0.93_wp,         &  !< parameter 147 - [W/(m*K)] thermal conductivity 4th wall layer (inside) roof
       1736000.0_wp,   &  !< parameter 148 - [J/(m3*K)] heat capacity 4th window layer (inside) roof
       0.19_wp         &  !< parameter 149 - [W/(m*K)] thermal conductivity 4th window layer (inside) roof
                        /)

    building_pars(:,5) = (/                                                                        &
       0.40_wp,        &  !< parameter 0   - [-] wall fraction above ground floor level
       0.60_wp,        &  !< parameter 1   - [-] window fraction above ground floor level
       0.0_wp,         &  !< parameter 2   - [-] green fraction above ground floor level
       0.0_wp,         &  !< parameter 3   - [-] green fraction roof above ground floor level
       2.0_wp,         &  !< parameter 4   - [m2/m2] LAI (Leaf Area Index) roof
       2.0_wp,         &  !< parameter 5   - [m2/m2] LAI (Leaf Area Index) on wall above ground floor level
       1520000.0_wp,   &  !< parameter 6   - [J/(m3*K)] heat capacity 1st wall layer (outside) above ground floor level
       2112000.0_wp,   &  !< parameter 7   - [J/(m3*K)] heat capacity 2nd wall layer above ground floor level
       2112000.0_wp,   &  !< parameter 8   - [J/(m3*K)] heat capacity 3rd wall layer above ground floor level
       0.93_wp,        &  !< parameter 9   - [W/(m*K)] thermal conductivity 1st wall layer (outside) above ground floor level
       2.10_wp,        &  !< parameter 10  - [W/(m*K)] thermal conductivity 2nd wall layer above ground floor level
       2.10_wp,        &  !< parameter 11  - [W/(m*K)] thermal conductivity 3rd wall layer above ground floor level
       301.15_wp,      &  !< parameter 12  - [K] indoor target summer temperature
       289.15_wp,      &  !< parameter 13  - [K] indoor target winter temperature
       0.93_wp,        &  !< parameter 14  - [-] wall emissivity above ground floor level
       0.86_wp,        &  !< parameter 15  - [-] green emissivity above ground floor level
       0.87_wp,        &  !< parameter 16  - [-] window emissivity above ground floor level
       0.65_wp,         &  !< parameter 17  - [-] window transmissivity (not visual transmissivity) above ground floor level
       0.001_wp,       &  !< parameter 18  - [m] z0 roughness above ground floor level
       0.0001_wp,      &  !< parameter 19  - [m] z0h/z0g roughness heat/humidity above ground floor level
       5.0_wp,         &  !< parameter 20  - [m] ground floor level height
       0.40_wp,        &  !< parameter 21  - [-] wall fraction ground floor level
       0.60_wp,        &  !< parameter 22  - [-] window fraction ground floor level
       0.0_wp,         &  !< parameter 23  - [-] green fraction ground floor level
       0.0_wp,         &  !< parameter 24  - [-] green fraction roof ground floor level
       2.0_wp,         &  !< parameter 25  - [m2/m2] LAI (Leaf Area Index) on wall ground floor level
       1520000.0_wp,   &  !< parameter 26  - [J/(m3*K)] heat capacity 1st wall layer (outside) ground floor level
       2112000.0_wp,   &  !< parameter 27  - [J/(m3*K)] heat capacity 2nd wall layer ground floor level
       2112000.0_wp,   &  !< parameter 28  - [J/(m3*K)] heat capacity 3rd wall layer ground floor level
       0.93_wp,        &  !< parameter 29  - [W/(m*K)] thermal conductivity 1st wall layer (outside) ground floor level
       2.10_wp,        &  !< parameter 30  - [W/(m*K)] thermal conductivity 2nd wall layer ground floor level
       2.10_wp,        &  !< parameter 31  - [W/(m*K)] thermal conductivity 3rd wall layer ground floor level
       0.93_wp,        &  !< parameter 32  - [-] wall emissivity ground floor level
       0.87_wp,        &  !< parameter 33  - [-] window emissivity ground floor level
       0.86_wp,        &  !< parameter 34  - [-] green emissivity ground floor level
       0.65_wp,         &  !< parameter 35  - [-] window transmissivity (not visual transmissivity) ground floor level
       0.001_wp,       &  !< parameter 36  - [m] z0 roughness ground floor level
       0.0001_wp,      &  !< parameter 37  - [m] z0h/z0q roughness heat/humidity
       36.0_wp,        &  !< parameter 38  - [-] wall albedo_type above ground floor level  (albedo_type specified in radiation model)
       5.0_wp,         &  !< parameter 39  - [-] green albedo_type above ground floor level  (albedo_type specified in radiation model)
       37.0_wp,        &  !< parameter 40  - [-] window albedo_type above ground floor level  (albedo_type specified in radiation model)
       0.02_wp,        &  !< parameter 41  - [m] 1st cumulative wall layer thickness above ground floor level
       0.17_wp,         &  !< parameter 42  - [m] 2nd cumulative wall layer thickness above ground floor level
       0.32_wp,        &  !< parameter 43  - [m] 3rd cumulative wall layer thickness above ground floor level
       0.34_wp,         &  !< parameter 44  - [m] 4th cumulative wall layer thickness above ground floor level
       20000.0_wp,     &  !< parameter 45  - [J/(m2*K)] heat capacity wall surface (1 cm air)
       23.0_wp,        &  !< parameter 46  - [W/(m2*K)] thermal conductivity of wall surface (1 cm air)
       20000.0_wp,     &  !< parameter 47  - [J/(m2*K)] heat capacity of window surface (1 cm air)
       20000.0_wp,     &  !< parameter 48  - [J/(m2*K)] heat capacity of green surface
       23.0_wp,        &  !< parameter 49  - [W/(m2*K)] thermal conductivity of window surface (1 cm air)
       10.0_wp,        &  !< parameter 50  - [W/(m2*K)] thermal conductivty of green surface
       1.0_wp,         &  !< parameter 51  - [-] wall fraction ground plate
       0.10_wp,        &  !< parameter 52  - [m] 1st cumulative wall layer thickness ground plate
       0.20_wp,        &  !< parameter 53  - [m] 2nd cumulative wall layer thickness ground plate
       0.26_wp,        &  !< parameter 54  - [m] 3rd cumulative wall layer thickness ground plate
       0.30_wp,        &  !< parameter 55  - [m] 4th cumulative wall layer thickness ground plate
       2112000.0_wp,   &  !< parameter 56  - [J/(m3*K)] heat capacity 1st wall layer (outside) ground plate
       2112000.0_wp,   &  !< parameter 57  - [J/(m3*K)] heat capacity 2nd wall layer ground plate
       2112000.0_wp,   &  !< parameter 58  - [J/(m3*K)] heat capacity 3rd wall layer ground plate
       2.10_wp,        &  !< parameter 59  - [W/(m*K)] thermal conductivity 1st wall layer (oustide) ground plate
       2.10_wp,        &  !< parameter 60  - [W/(m*K)] thermal conductivity 2nd wall layer ground plate
       2.1_wp,         &  !< parameter 61  - [W/(m*K)] thermal conductivity 3rd wall layer ground plate
       0.02_wp,        &  !< parameter 62  - [m] 1st cumulative wall layer thickness ground floor level
       0.17_wp,         &  !< parameter 63  - [m] 2nd cumulative wall layer thickness ground floor level
       0.32_wp,        &  !< parameter 64  - [m] 3rd cumulative wall layer thickness ground floor level
       0.34_wp,         &  !< parameter 65  - [m] 4th cumulative wall layer thickness ground floor level
       36.0_wp,        &  !< parameter 66  - [-] wall albedo_type ground floor level (albedo_type specified in radiation model)
       0.02_wp,        &  !< parameter 67  - [m] 1st cumulative window layer thickness ground floor level
       0.04_wp,        &  !< parameter 68  - [m] 2nd cumulative window layer thickness ground floor level
       0.06_wp,        &  !< parameter 69  - [m] 3rd cumulative window layer thickness ground floor level
       0.08_wp,        &  !< parameter 70  - [m] 4th cumulative window layer thickness ground floor level
       1736000.0_wp,   &  !< parameter 71  - [J/(m3*K)] heat capacity 1st window layer (outside) ground floor level
       1736000.0_wp,   &  !< parameter 72  - [J/(m3*K)] heat capacity 2nd window layer ground floor level
       1736000.0_wp,   &  !< parameter 73  - [J/(m3*K)] heat capacity 3rd window layer ground floor level
       0.19_wp,        &  !< parameter 74  - [W/(m*K)] thermal conductivity 1st window layer (outside) ground floor level
       0.19_wp,        &  !< parameter 75  - [W/(m*K)] thermal conductivity 2nd window layer ground floor level
       0.19_wp,        &  !< parameter 76  - [W/(m*K)] thermal conductivity 3rd window layer ground floor level
       37.0_wp,        &  !< parameter 77  - [-] window albedo_type ground floor level (albedo_type specified in radiation model)
       5.0_wp,         &  !< parameter 78  - [-] green albedo_type ground floor level (albedo_type specified in radiation model)
       0.02_wp,        &  !< parameter 79  - [m] 1st cumulative window layer thickness above ground floor level
       0.04_wp,        &  !< parameter 80  - [m] 2nd thickness window layer above ground floor level
       0.06_wp,        &  !< parameter 81  - [m] 3rd cumulative window layer thickness above ground floor level
       0.08_wp,        &  !< parameter 82  - [m] 4th cumulative window layer thickness above ground floor level
       1736000.0_wp,   &  !< parameter 83  - [J/(m3*K)] heat capacity 1st window layer (outside) above ground floor level
       1736000.0_wp,   &  !< parameter 84  - [J/(m3*K)] heat capacity 2nd window layer above ground floor level
       1736000.0_wp,   &  !< parameter 85  - [J/(m3*K)] heat capacity 3rd window layer above ground floor level
       0.19_wp,        &  !< parameter 86  - [W/(m*K)] thermal conductivity 1st window layer (outside) above ground floor level
       0.19_wp,        &  !< parameter 86  - [W/(m*K)] thermal conductivity 2nd window layer above ground floor level
       0.19_wp,        &  !< parameter 87  - [W/(m*K)] thermal conductivity 3rd window layer above ground floor level
       1.0_wp,         &  !< parameter 89  - [-] wall fraction roof
       0.02_wp,        &  !< parameter 90  - [m] 1st cumulative wall layer thickness roof
       0.17_wp,        &  !< parameter 91  - [m] 2nd cumulative wall layer thickness roof
       0.32_wp,        &  !< parameter 92  - [m] 3rd cumulative wall layer thickness roof
       0.34_wp,         &  !< parameter 93  - [m] 4th cumulative wall layer thickness roof
       1520000.0_wp,   &  !< parameter 94  - [J/(m3*K)] heat capacity 1st wall layer (outside) roof
       2112000.0_wp,    &  !< parameter 95  - [J/(m3*K)] heat capacity 2nd wall layer roof
       2112000.0_wp,    &  !< parameter 96  - [J/(m3*K)] heat capacity 3rd wall layer roof
       0.93_wp,        &  !< parameter 97  - [W/(m*K)] thermal conductivity 1st wall layer (outside) roof
       2.10_wp,        &  !< parameter 98  - [W/(m*K)] thermal conductivity 2nd wall layer roof
       2.10_wp,        &  !< parameter 99  - [W/(m*K)] thermal conductivity 3rd wall layer roof
       0.93_wp,        &  !< parameter 100 - [-] wall emissivity roof
       42.0_wp,        &  !< parameter 101 - [-] wall albedo_type roof (albedo_type specified in radiation model)
       0.0_wp,         &  !< parameter 102 - [-] window fraction roof
       0.02_wp,        &  !< parameter 103 - [m] window 1st layer thickness roof
       0.04_wp,        &  !< parameter 104 - [m] window 2nd layer thickness roof
       0.06_wp,        &  !< parameter 105 - [m] window 3rd layer thickness roof
       0.08_wp,        &  !< parameter 106 - [m] window 4th layer thickness roof
       1736000.0_wp,   &  !< parameter 107 - [J/(m3*K)] heat capacity 1st window layer (outside) roof
       1736000.0_wp,   &  !< parameter 108 - [J/(m3*K)] heat capacity 2nd window layer roof
       1736000.0_wp,   &  !< parameter 109 - [J/(m3*K)] heat capacity 3rd window layer roof
       0.19_wp,        &  !< parameter 110 - [W/(m*K)] thermal conductivity 1st window layer (outside) roof
       0.19_wp,        &  !< parameter 111 - [W/(m*K)] thermal conductivity 2nd window layer roof
       0.19_wp,        &  !< parameter 112 - [W/(m*K)] thermal conductivity 3rd window layer roof
       0.87_wp,        &  !< parameter 113 - [-] window emissivity roof
       0.65_wp,         &  !< parameter 114 - [-] window transmissivity (not visual transmissivity) roof
       37.0_wp,        &  !< parameter 115 - [-] window albedo_type roof (albedo_type specified in radiation model)
       0.86_wp,        &  !< parameter 116 - [-] green emissivity roof
       5.0_wp,         &  !< parameter 117 - [-] green albedo_type roof (albedo_type specified in radiation model)
       0.0_wp,         &  !< parameter 118 - [-] green type roof
       0.75_wp,        &  !< parameter 119 - [-] shading factor
       0.70_wp,         &  !< parameter 120 - [-] g-value windows
       1.7_wp,         &  !< parameter 121 - [W/(m2*K)] u-value windows
       0.5_wp,         &  !< parameter 122 - [1/h] basic airflow without occupancy of the room for - summer 0.5_wp, winter 0.5
       2.0_wp,         &  !< parameter 123 - [1/h] additional airflow dependent on occupancy of the room for - summer 1.5_wp, winter 0.0
       0.6_wp,         &  !< parameter 124 - [-] heat recovery efficiency
       2.5_wp,         &  !< parameter 125 - [m2/m2] dynamic parameter specific effective surface
       165000.0_wp,    &  !< parameter 126 - [J/(m2*K)] dynamic parameter innner heat storage
       4.5_wp,         &  !< parameter 127 - [m2/m2] ratio internal surface/floor area
       0.0_wp,       &  !< parameter 128 - [W] maximal heating capacity
       -80.0_wp,         &  !< parameter 129 - [W] maximal cooling capacity
       5.0_wp,         &  !< parameter 130 - [W/m2] additional internal heat gains dependent on occupancy of the room
       2.1_wp,         &  !< parameter 131 - [W/m2] basic internal heat gains without occupancy of the room
       5.0_wp,         &  !< parameter 132 - [m] storey height
       0.2_wp,         &  !< parameter 133 - [m] ceiling construction height
       0.0_wp,         &  !< parameter 134 - [-] anthropogenic heat output for heating
       1.0_wp,       &  !< parameter 135 - [-] anthropogenic heat output for cooling
       1526000.0_wp,   &  !< parameter 136 - [J/(m3*K)] heat capacity 4th wall layer (inside) above ground floor level
       0.93_wp,         &  !< parameter 137 - [W/(m*K)] thermal conductivity 4th wall layer (inside) above ground floor level
       1526000.0_wp,   &  !< parameter 138 - [J/(m3*K)] capacity 4th wall layer (inside) ground floor level
       0.93_wp,         &  !< parameter 139 - [W/(m*K)] thermal conductivity 4th wall layer (inside) ground floor level
       709650.0_wp,    &  !< parameter 140 - [J/(m3*K)] heat capacity 4th wall layer (inside) ground plate
       0.12_wp,        &  !< parameter 141 - [W/(m*K)] thermal conductivity 4th wall layer (inside) ground plate
       1736000.0_wp,   &  !< parameter 142 - [J/(m3*K)] heat capacity 4th window layer (inside) ground floor level
       0.19_wp,        &  !< parameter 143 - [W/(m*K)] thermal conductivity 4th window layer (inside) ground floor level
       1736000.0_wp,   &  !< parameter 144 - [J/(m3*K)] heat capacity 4th layer (inside) above ground floor level
       0.45_wp,        &  !< parameter 145 - [W/(m*K)] thermal conductivity 4th window layer (inside) above ground floor level
       1526000.0_wp,   &  !< parameter 146 - [J/(m3*K)] heat capacity 4th wall layer (inside) roof
       0.93_wp,         &  !< parameter 147 - [W/(m*K)] thermal conductivity 4th wall layer (inside) roof
       1736000.0_wp,   &  !< parameter 148 - [J/(m3*K)] heat capacity 4th window layer (inside) roof
       0.19_wp         &  !< parameter 149 - [W/(m*K)] thermal conductivity 4th window layer (inside) roof
                        /)

    building_pars(:,6) = (/                                                                        &
       0.40_wp,        &  !< parameter 0   - [-] wall fraction above ground floor level
       0.60_wp,        &  !< parameter 1   - [-] window fraction above ground floor level
       0.0_wp,         &  !< parameter 2   - [-] green fraction above ground floor level
       0.0_wp,         &  !< parameter 3   - [-] green fraction roof above ground floor level
       2.0_wp,         &  !< parameter 4   - [m2/m2] LAI (Leaf Area Index) roof
       2.0_wp,         &  !< parameter 5   - [m2/m2] LAI (Leaf Area Index) on wall above ground floor level
       1520000.0_wp,   &  !< parameter 6   - [J/(m3*K)] heat capacity 1st wall layer (outside) above ground floor level
       2112000.0_wp,   &  !< parameter 7   - [J/(m3*K)] heat capacity 2nd wall layer above ground floor level
       2112000.0_wp,   &  !< parameter 8   - [J/(m3*K)] heat capacity 3rd wall layer above ground floor level
       0.93_wp,        &  !< parameter 9   - [W/(m*K)] thermal conductivity 1st wall layer (outside) above ground floor level
       2.10_wp,        &  !< parameter 10  - [W/(m*K)] thermal conductivity 2nd wall layer above ground floor level
       2.10_wp,        &  !< parameter 11  - [W/(m*K)] thermal conductivity 3rd wall layer above ground floor level
       301.15_wp,      &  !< parameter 12  - [K] indoor target summer temperature
       289.15_wp,      &  !< parameter 13  - [K] indoor target winter temperature
       0.93_wp,        &  !< parameter 14  - [-] wall emissivity above ground floor level
       0.86_wp,        &  !< parameter 15  - [-] green emissivity above ground floor level
       0.87_wp,        &  !< parameter 16  - [-] window emissivity above ground floor level
       0.65_wp,         &  !< parameter 17  - [-] window transmissivity (not visual transmissivity) above ground floor level
       0.001_wp,       &  !< parameter 18  - [m] z0 roughness above ground floor level
       0.0001_wp,      &  !< parameter 19  - [m] z0h/z0g roughness heat/humidity above ground floor level
       5.0_wp,         &  !< parameter 20  - [m] ground floor level height
       0.40_wp,        &  !< parameter 21  - [-] wall fraction ground floor level
       0.60_wp,        &  !< parameter 22  - [-] window fraction ground floor level
       0.0_wp,         &  !< parameter 23  - [-] green fraction ground floor level
       0.0_wp,         &  !< parameter 24  - [-] green fraction roof ground floor level
       2.0_wp,         &  !< parameter 25  - [m2/m2] LAI (Leaf Area Index) on wall ground floor level
       1520000.0_wp,   &  !< parameter 26  - [J/(m3*K)] heat capacity 1st wall layer (outside) ground floor level
       2112000.0_wp,   &  !< parameter 27  - [J/(m3*K)] heat capacity 2nd wall layer ground floor level
       2112000.0_wp,   &  !< parameter 28  - [J/(m3*K)] heat capacity 3rd wall layer ground floor level
       0.93_wp,        &  !< parameter 29  - [W/(m*K)] thermal conductivity 1st wall layer (outside) ground floor level
       2.10_wp,        &  !< parameter 30  - [W/(m*K)] thermal conductivity 2nd wall layer ground floor level
       2.10_wp,        &  !< parameter 31  - [W/(m*K)] thermal conductivity 3rd wall layer ground floor level
       0.93_wp,        &  !< parameter 32  - [-] wall emissivity ground floor level
       0.87_wp,        &  !< parameter 33  - [-] window emissivity ground floor level
       0.86_wp,        &  !< parameter 34  - [-] green emissivity ground floor level
       0.65_wp,         &  !< parameter 35  - [-] window transmissivity (not visual transmissivity) ground floor level
       0.001_wp,       &  !< parameter 36  - [m] z0 roughness ground floor level
       0.0001_wp,      &  !< parameter 37  - [m] z0h/z0q roughness heat/humidity
       36.0_wp,        &  !< parameter 38  - [-] wall albedo_type above ground floor level  (albedo_type specified in radiation model)
       5.0_wp,         &  !< parameter 39  - [-] green albedo_type above ground floor level  (albedo_type specified in radiation model)
       37.0_wp,        &  !< parameter 40  - [-] window albedo_type above ground floor level  (albedo_type specified in radiation model)
       0.02_wp,        &  !< parameter 41  - [m] 1st cumulative wall layer thickness above ground floor level
       0.17_wp,         &  !< parameter 42  - [m] 2nd cumulative wall layer thickness above ground floor level
       0.32_wp,        &  !< parameter 43  - [m] 3rd cumulative wall layer thickness above ground floor level
       0.34_wp,         &  !< parameter 44  - [m] 4th cumulative wall layer thickness above ground floor level
       20000.0_wp,     &  !< parameter 45  - [J/(m2*K)] heat capacity wall surface (1 cm air)
       23.0_wp,        &  !< parameter 46  - [W/(m2*K)] thermal conductivity of wall surface (1 cm air)
       20000.0_wp,     &  !< parameter 47  - [J/(m2*K)] heat capacity of window surface (1 cm air)
       20000.0_wp,     &  !< parameter 48  - [J/(m2*K)] heat capacity of green surface
       23.0_wp,        &  !< parameter 49  - [W/(m2*K)] thermal conductivity of window surface (1 cm air)
       10.0_wp,        &  !< parameter 50  - [W/(m2*K)] thermal conductivty of green surface
       1.0_wp,         &  !< parameter 51  - [-] wall fraction ground plate
       0.10_wp,        &  !< parameter 52  - [m] 1st cumulative wall layer thickness ground plate
       0.20_wp,        &  !< parameter 53  - [m] 2nd cumulative wall layer thickness ground plate
       0.26_wp,        &  !< parameter 54  - [m] 3rd cumulative wall layer thickness ground plate
       0.30_wp,        &  !< parameter 55  - [m] 4th cumulative wall layer thickness ground plate
       2112000.0_wp,   &  !< parameter 56  - [J/(m3*K)] heat capacity 1st wall layer (outside) ground plate
       2112000.0_wp,   &  !< parameter 57  - [J/(m3*K)] heat capacity 2nd wall layer ground plate
       2112000.0_wp,   &  !< parameter 58  - [J/(m3*K)] heat capacity 3rd wall layer ground plate
       2.10_wp,        &  !< parameter 59  - [W/(m*K)] thermal conductivity 1st wall layer (oustide) ground plate
       2.10_wp,        &  !< parameter 60  - [W/(m*K)] thermal conductivity 2nd wall layer ground plate
       2.1_wp,         &  !< parameter 61  - [W/(m*K)] thermal conductivity 3rd wall layer ground plate
       0.02_wp,        &  !< parameter 62  - [m] 1st cumulative wall layer thickness ground floor level
       0.17_wp,         &  !< parameter 63  - [m] 2nd cumulative wall layer thickness ground floor level
       0.32_wp,        &  !< parameter 64  - [m] 3rd cumulative wall layer thickness ground floor level
       0.34_wp,         &  !< parameter 65  - [m] 4th cumulative wall layer thickness ground floor level
       36.0_wp,        &  !< parameter 66  - [-] wall albedo_type ground floor level (albedo_type specified in radiation model)
       0.02_wp,        &  !< parameter 67  - [m] 1st cumulative window layer thickness ground floor level
       0.04_wp,        &  !< parameter 68  - [m] 2nd cumulative window layer thickness ground floor level
       0.06_wp,        &  !< parameter 69  - [m] 3rd cumulative window layer thickness ground floor level
       0.08_wp,        &  !< parameter 70  - [m] 4th cumulative window layer thickness ground floor level
       1736000.0_wp,   &  !< parameter 71  - [J/(m3*K)] heat capacity 1st window layer (outside) ground floor level
       1736000.0_wp,   &  !< parameter 72  - [J/(m3*K)] heat capacity 2nd window layer ground floor level
       1736000.0_wp,   &  !< parameter 73  - [J/(m3*K)] heat capacity 3rd window layer ground floor level
       0.19_wp,        &  !< parameter 74  - [W/(m*K)] thermal conductivity 1st window layer (outside) ground floor level
       0.19_wp,        &  !< parameter 75  - [W/(m*K)] thermal conductivity 2nd window layer ground floor level
       0.19_wp,        &  !< parameter 76  - [W/(m*K)] thermal conductivity 3rd window layer ground floor level
       37.0_wp,        &  !< parameter 77  - [-] window albedo_type ground floor level (albedo_type specified in radiation model)
       5.0_wp,         &  !< parameter 78  - [-] green albedo_type ground floor level (albedo_type specified in radiation model)
       0.02_wp,        &  !< parameter 79  - [m] 1st cumulative window layer thickness above ground floor level
       0.04_wp,        &  !< parameter 80  - [m] 2nd thickness window layer above ground floor level
       0.06_wp,        &  !< parameter 81  - [m] 3rd cumulative window layer thickness above ground floor level
       0.08_wp,        &  !< parameter 82  - [m] 4th cumulative window layer thickness above ground floor level
       1736000.0_wp,   &  !< parameter 83  - [J/(m3*K)] heat capacity 1st window layer (outside) above ground floor level
       1736000.0_wp,   &  !< parameter 84  - [J/(m3*K)] heat capacity 2nd window layer above ground floor level
       1736000.0_wp,   &  !< parameter 85  - [J/(m3*K)] heat capacity 3rd window layer above ground floor level
       0.19_wp,        &  !< parameter 86  - [W/(m*K)] thermal conductivity 1st window layer (outside) above ground floor level
       0.19_wp,        &  !< parameter 86  - [W/(m*K)] thermal conductivity 2nd window layer above ground floor level
       0.19_wp,        &  !< parameter 87  - [W/(m*K)] thermal conductivity 3rd window layer above ground floor level
       1.0_wp,         &  !< parameter 89  - [-] wall fraction roof
       0.02_wp,        &  !< parameter 90  - [m] 1st cumulative wall layer thickness roof
       0.17_wp,        &  !< parameter 91  - [m] 2nd cumulative wall layer thickness roof
       0.32_wp,        &  !< parameter 92  - [m] 3rd cumulative wall layer thickness roof
       0.34_wp,         &  !< parameter 93  - [m] 4th cumulative wall layer thickness roof
       1520000.0_wp,   &  !< parameter 94  - [J/(m3*K)] heat capacity 1st wall layer (outside) roof
       2112000.0_wp,    &  !< parameter 95  - [J/(m3*K)] heat capacity 2nd wall layer roof
       2112000.0_wp,    &  !< parameter 96  - [J/(m3*K)] heat capacity 3rd wall layer roof
       0.93_wp,        &  !< parameter 97  - [W/(m*K)] thermal conductivity 1st wall layer (outside) roof
       2.10_wp,        &  !< parameter 98  - [W/(m*K)] thermal conductivity 2nd wall layer roof
       2.10_wp,        &  !< parameter 99  - [W/(m*K)] thermal conductivity 3rd wall layer roof
       0.93_wp,        &  !< parameter 100 - [-] wall emissivity roof
       42.0_wp,        &  !< parameter 101 - [-] wall albedo_type roof (albedo_type specified in radiation model)
       0.0_wp,         &  !< parameter 102 - [-] window fraction roof
       0.02_wp,        &  !< parameter 103 - [m] window 1st layer thickness roof
       0.04_wp,        &  !< parameter 104 - [m] window 2nd layer thickness roof
       0.06_wp,        &  !< parameter 105 - [m] window 3rd layer thickness roof
       0.08_wp,        &  !< parameter 106 - [m] window 4th layer thickness roof
       1736000.0_wp,   &  !< parameter 107 - [J/(m3*K)] heat capacity 1st window layer (outside) roof
       1736000.0_wp,   &  !< parameter 108 - [J/(m3*K)] heat capacity 2nd window layer roof
       1736000.0_wp,   &  !< parameter 109 - [J/(m3*K)] heat capacity 3rd window layer roof
       0.19_wp,        &  !< parameter 110 - [W/(m*K)] thermal conductivity 1st window layer (outside) roof
       0.19_wp,        &  !< parameter 111 - [W/(m*K)] thermal conductivity 2nd window layer roof
       0.19_wp,        &  !< parameter 112 - [W/(m*K)] thermal conductivity 3rd window layer roof
       0.87_wp,        &  !< parameter 113 - [-] window emissivity roof
       0.65_wp,         &  !< parameter 114 - [-] window transmissivity (not visual transmissivity) roof
       37.0_wp,        &  !< parameter 115 - [-] window albedo_type roof (albedo_type specified in radiation model)
       0.86_wp,        &  !< parameter 116 - [-] green emissivity roof
       5.0_wp,         &  !< parameter 117 - [-] green albedo_type roof (albedo_type specified in radiation model)
       0.0_wp,         &  !< parameter 118 - [-] green type roof
       0.75_wp,        &  !< parameter 119 - [-] shading factor
       0.70_wp,         &  !< parameter 120 - [-] g-value windows
       1.7_wp,         &  !< parameter 121 - [W/(m2*K)] u-value windows
       0.5_wp,         &  !< parameter 122 - [1/h] basic airflow without occupancy of the room for - summer 0.5_wp, winter 0.5
       2.0_wp,         &  !< parameter 123 - [1/h] additional airflow dependent on occupancy of the room for - summer 1.5_wp, winter 0.0
       0.6_wp,         &  !< parameter 124 - [-] heat recovery efficiency
       2.5_wp,         &  !< parameter 125 - [m2/m2] dynamic parameter specific effective surface
       165000.0_wp,    &  !< parameter 126 - [J/(m2*K)] dynamic parameter innner heat storage
       4.5_wp,         &  !< parameter 127 - [m2/m2] ratio internal surface/floor area
       0.0_wp,       &  !< parameter 128 - [W] maximal heating capacity
       -80.0_wp,         &  !< parameter 129 - [W] maximal cooling capacity
       5.0_wp,         &  !< parameter 130 - [W/m2] additional internal heat gains dependent on occupancy of the room
       2.1_wp,         &  !< parameter 131 - [W/m2] basic internal heat gains without occupancy of the room
       5.0_wp,         &  !< parameter 132 - [m] storey height
       0.2_wp,         &  !< parameter 133 - [m] ceiling construction height
       0.0_wp,         &  !< parameter 134 - [-] anthropogenic heat output for heating
       1.0_wp,       &  !< parameter 135 - [-] anthropogenic heat output for cooling
       1526000.0_wp,   &  !< parameter 136 - [J/(m3*K)] heat capacity 4th wall layer (inside) above ground floor level
       0.93_wp,         &  !< parameter 137 - [W/(m*K)] thermal conductivity 4th wall layer (inside) above ground floor level
       1526000.0_wp,   &  !< parameter 138 - [J/(m3*K)] capacity 4th wall layer (inside) ground floor level
       0.93_wp,         &  !< parameter 139 - [W/(m*K)] thermal conductivity 4th wall layer (inside) ground floor level
       709650.0_wp,    &  !< parameter 140 - [J/(m3*K)] heat capacity 4th wall layer (inside) ground plate
       0.12_wp,        &  !< parameter 141 - [W/(m*K)] thermal conductivity 4th wall layer (inside) ground plate
       1736000.0_wp,   &  !< parameter 142 - [J/(m3*K)] heat capacity 4th window layer (inside) ground floor level
       0.19_wp,        &  !< parameter 143 - [W/(m*K)] thermal conductivity 4th window layer (inside) ground floor level
       1736000.0_wp,   &  !< parameter 144 - [J/(m3*K)] heat capacity 4th layer (inside) above ground floor level
       0.45_wp,        &  !< parameter 145 - [W/(m*K)] thermal conductivity 4th window layer (inside) above ground floor level
       1526000.0_wp,   &  !< parameter 146 - [J/(m3*K)] heat capacity 4th wall layer (inside) roof
       0.93_wp,         &  !< parameter 147 - [W/(m*K)] thermal conductivity 4th wall layer (inside) roof
       1736000.0_wp,   &  !< parameter 148 - [J/(m3*K)] heat capacity 4th window layer (inside) roof
       0.19_wp         &  !< parameter 149 - [W/(m*K)] thermal conductivity 4th window layer (inside) roof
                        /)

    building_pars(:,7) = (/                                                                        &
       0.40_wp,        &  !< parameter 0   - [-] wall fraction above ground floor level
       0.60_wp,        &  !< parameter 1   - [-] window fraction above ground floor level
       0.0_wp,         &  !< parameter 2   - [-] green fraction above ground floor level
       0.0_wp,         &  !< parameter 3   - [-] green fraction roof above ground floor level
       2.0_wp,         &  !< parameter 4   - [m2/m2] LAI (Leaf Area Index) roof
       2.0_wp,         &  !< parameter 5   - [m2/m2] LAI (Leaf Area Index) on wall above ground floor level
       1520000.0_wp,   &  !< parameter 6   - [J/(m3*K)] heat capacity 1st wall layer (outside) above ground floor level
       2112000.0_wp,   &  !< parameter 7   - [J/(m3*K)] heat capacity 2nd wall layer above ground floor level
       2112000.0_wp,   &  !< parameter 8   - [J/(m3*K)] heat capacity 3rd wall layer above ground floor level
       0.93_wp,        &  !< parameter 9   - [W/(m*K)] thermal conductivity 1st wall layer (outside) above ground floor level
       2.10_wp,        &  !< parameter 10  - [W/(m*K)] thermal conductivity 2nd wall layer above ground floor level
       2.10_wp,        &  !< parameter 11  - [W/(m*K)] thermal conductivity 3rd wall layer above ground floor level
       301.15_wp,      &  !< parameter 12  - [K] indoor target summer temperature
       289.15_wp,      &  !< parameter 13  - [K] indoor target winter temperature
       0.93_wp,        &  !< parameter 14  - [-] wall emissivity above ground floor level
       0.86_wp,        &  !< parameter 15  - [-] green emissivity above ground floor level
       0.87_wp,        &  !< parameter 16  - [-] window emissivity above ground floor level
       0.65_wp,         &  !< parameter 17  - [-] window transmissivity (not visual transmissivity) above ground floor level
       0.001_wp,       &  !< parameter 18  - [m] z0 roughness above ground floor level
       0.0001_wp,      &  !< parameter 19  - [m] z0h/z0g roughness heat/humidity above ground floor level
       5.0_wp,         &  !< parameter 20  - [m] ground floor level height
       0.40_wp,        &  !< parameter 21  - [-] wall fraction ground floor level
       0.60_wp,        &  !< parameter 22  - [-] window fraction ground floor level
       0.0_wp,         &  !< parameter 23  - [-] green fraction ground floor level
       0.0_wp,         &  !< parameter 24  - [-] green fraction roof ground floor level
       2.0_wp,         &  !< parameter 25  - [m2/m2] LAI (Leaf Area Index) on wall ground floor level
       1520000.0_wp,   &  !< parameter 26  - [J/(m3*K)] heat capacity 1st wall layer (outside) ground floor level
       2112000.0_wp,   &  !< parameter 27  - [J/(m3*K)] heat capacity 2nd wall layer ground floor level
       2112000.0_wp,   &  !< parameter 28  - [J/(m3*K)] heat capacity 3rd wall layer ground floor level
       0.93_wp,        &  !< parameter 29  - [W/(m*K)] thermal conductivity 1st wall layer (outside) ground floor level
       2.10_wp,        &  !< parameter 30  - [W/(m*K)] thermal conductivity 2nd wall layer ground floor level
       2.10_wp,        &  !< parameter 31  - [W/(m*K)] thermal conductivity 3rd wall layer ground floor level
       0.93_wp,        &  !< parameter 32  - [-] wall emissivity ground floor level
       0.87_wp,        &  !< parameter 33  - [-] window emissivity ground floor level
       0.86_wp,        &  !< parameter 34  - [-] green emissivity ground floor level
       0.65_wp,         &  !< parameter 35  - [-] window transmissivity (not visual transmissivity) ground floor level
       0.001_wp,       &  !< parameter 36  - [m] z0 roughness ground floor level
       0.0001_wp,      &  !< parameter 37  - [m] z0h/z0q roughness heat/humidity
       36.0_wp,        &  !< parameter 38  - [-] wall albedo_type above ground floor level  (albedo_type specified in radiation model)
       5.0_wp,         &  !< parameter 39  - [-] green albedo_type above ground floor level  (albedo_type specified in radiation model)
       37.0_wp,        &  !< parameter 40  - [-] window albedo_type above ground floor level  (albedo_type specified in radiation model)
       0.02_wp,        &  !< parameter 41  - [m] 1st cumulative wall layer thickness above ground floor level
       0.17_wp,         &  !< parameter 42  - [m] 2nd cumulative wall layer thickness above ground floor level
       0.32_wp,        &  !< parameter 43  - [m] 3rd cumulative wall layer thickness above ground floor level
       0.34_wp,         &  !< parameter 44  - [m] 4th cumulative wall layer thickness above ground floor level
       20000.0_wp,     &  !< parameter 45  - [J/(m2*K)] heat capacity wall surface (1 cm air)
       23.0_wp,        &  !< parameter 46  - [W/(m2*K)] thermal conductivity of wall surface (1 cm air)
       20000.0_wp,     &  !< parameter 47  - [J/(m2*K)] heat capacity of window surface (1 cm air)
       20000.0_wp,     &  !< parameter 48  - [J/(m2*K)] heat capacity of green surface
       23.0_wp,        &  !< parameter 49  - [W/(m2*K)] thermal conductivity of window surface (1 cm air)
       10.0_wp,        &  !< parameter 50  - [W/(m2*K)] thermal conductivty of green surface
       1.0_wp,         &  !< parameter 51  - [-] wall fraction ground plate
       0.10_wp,        &  !< parameter 52  - [m] 1st cumulative wall layer thickness ground plate
       0.20_wp,        &  !< parameter 53  - [m] 2nd cumulative wall layer thickness ground plate
       0.26_wp,        &  !< parameter 54  - [m] 3rd cumulative wall layer thickness ground plate
       0.30_wp,        &  !< parameter 55  - [m] 4th cumulative wall layer thickness ground plate
       2112000.0_wp,   &  !< parameter 56  - [J/(m3*K)] heat capacity 1st wall layer (outside) ground plate
       2112000.0_wp,   &  !< parameter 57  - [J/(m3*K)] heat capacity 2nd wall layer ground plate
       2112000.0_wp,   &  !< parameter 58  - [J/(m3*K)] heat capacity 3rd wall layer ground plate
       2.10_wp,        &  !< parameter 59  - [W/(m*K)] thermal conductivity 1st wall layer (oustide) ground plate
       2.10_wp,        &  !< parameter 60  - [W/(m*K)] thermal conductivity 2nd wall layer ground plate
       2.1_wp,         &  !< parameter 61  - [W/(m*K)] thermal conductivity 3rd wall layer ground plate
       0.02_wp,        &  !< parameter 62  - [m] 1st cumulative wall layer thickness ground floor level
       0.17_wp,         &  !< parameter 63  - [m] 2nd cumulative wall layer thickness ground floor level
       0.32_wp,        &  !< parameter 64  - [m] 3rd cumulative wall layer thickness ground floor level
       0.34_wp,         &  !< parameter 65  - [m] 4th cumulative wall layer thickness ground floor level
       36.0_wp,        &  !< parameter 66  - [-] wall albedo_type ground floor level (albedo_type specified in radiation model)
       0.02_wp,        &  !< parameter 67  - [m] 1st cumulative window layer thickness ground floor level
       0.04_wp,        &  !< parameter 68  - [m] 2nd cumulative window layer thickness ground floor level
       0.06_wp,        &  !< parameter 69  - [m] 3rd cumulative window layer thickness ground floor level
       0.08_wp,        &  !< parameter 70  - [m] 4th cumulative window layer thickness ground floor level
       1736000.0_wp,   &  !< parameter 71  - [J/(m3*K)] heat capacity 1st window layer (outside) ground floor level
       1736000.0_wp,   &  !< parameter 72  - [J/(m3*K)] heat capacity 2nd window layer ground floor level
       1736000.0_wp,   &  !< parameter 73  - [J/(m3*K)] heat capacity 3rd window layer ground floor level
       0.19_wp,        &  !< parameter 74  - [W/(m*K)] thermal conductivity 1st window layer (outside) ground floor level
       0.19_wp,        &  !< parameter 75  - [W/(m*K)] thermal conductivity 2nd window layer ground floor level
       0.19_wp,        &  !< parameter 76  - [W/(m*K)] thermal conductivity 3rd window layer ground floor level
       37.0_wp,        &  !< parameter 77  - [-] window albedo_type ground floor level (albedo_type specified in radiation model)
       5.0_wp,         &  !< parameter 78  - [-] green albedo_type ground floor level (albedo_type specified in radiation model)
       0.02_wp,        &  !< parameter 79  - [m] 1st cumulative window layer thickness above ground floor level
       0.04_wp,        &  !< parameter 80  - [m] 2nd thickness window layer above ground floor level
       0.06_wp,        &  !< parameter 81  - [m] 3rd cumulative window layer thickness above ground floor level
       0.08_wp,        &  !< parameter 82  - [m] 4th cumulative window layer thickness above ground floor level
       1736000.0_wp,   &  !< parameter 83  - [J/(m3*K)] heat capacity 1st window layer (outside) above ground floor level
       1736000.0_wp,   &  !< parameter 84  - [J/(m3*K)] heat capacity 2nd window layer above ground floor level
       1736000.0_wp,   &  !< parameter 85  - [J/(m3*K)] heat capacity 3rd window layer above ground floor level
       0.19_wp,        &  !< parameter 86  - [W/(m*K)] thermal conductivity 1st window layer (outside) above ground floor level
       0.19_wp,        &  !< parameter 86  - [W/(m*K)] thermal conductivity 2nd window layer above ground floor level
       0.19_wp,        &  !< parameter 87  - [W/(m*K)] thermal conductivity 3rd window layer above ground floor level
       1.0_wp,         &  !< parameter 89  - [-] wall fraction roof
       0.02_wp,        &  !< parameter 90  - [m] 1st cumulative wall layer thickness roof
       0.17_wp,        &  !< parameter 91  - [m] 2nd cumulative wall layer thickness roof
       0.32_wp,        &  !< parameter 92  - [m] 3rd cumulative wall layer thickness roof
       0.34_wp,         &  !< parameter 93  - [m] 4th cumulative wall layer thickness roof
       1520000.0_wp,   &  !< parameter 94  - [J/(m3*K)] heat capacity 1st wall layer (outside) roof
       2112000.0_wp,    &  !< parameter 95  - [J/(m3*K)] heat capacity 2nd wall layer roof
       2112000.0_wp,    &  !< parameter 96  - [J/(m3*K)] heat capacity 3rd wall layer roof
       0.93_wp,        &  !< parameter 97  - [W/(m*K)] thermal conductivity 1st wall layer (outside) roof
       2.10_wp,        &  !< parameter 98  - [W/(m*K)] thermal conductivity 2nd wall layer roof
       2.10_wp,        &  !< parameter 99  - [W/(m*K)] thermal conductivity 3rd wall layer roof
       0.93_wp,        &  !< parameter 100 - [-] wall emissivity roof
       42.0_wp,        &  !< parameter 101 - [-] wall albedo_type roof (albedo_type specified in radiation model)
       0.0_wp,         &  !< parameter 102 - [-] window fraction roof
       0.02_wp,        &  !< parameter 103 - [m] window 1st layer thickness roof
       0.04_wp,        &  !< parameter 104 - [m] window 2nd layer thickness roof
       0.06_wp,        &  !< parameter 105 - [m] window 3rd layer thickness roof
       0.08_wp,        &  !< parameter 106 - [m] window 4th layer thickness roof
       1736000.0_wp,   &  !< parameter 107 - [J/(m3*K)] heat capacity 1st window layer (outside) roof
       1736000.0_wp,   &  !< parameter 108 - [J/(m3*K)] heat capacity 2nd window layer roof
       1736000.0_wp,   &  !< parameter 109 - [J/(m3*K)] heat capacity 3rd window layer roof
       0.19_wp,        &  !< parameter 110 - [W/(m*K)] thermal conductivity 1st window layer (outside) roof
       0.19_wp,        &  !< parameter 111 - [W/(m*K)] thermal conductivity 2nd window layer roof
       0.19_wp,        &  !< parameter 112 - [W/(m*K)] thermal conductivity 3rd window layer roof
       0.87_wp,        &  !< parameter 113 - [-] window emissivity roof
       0.65_wp,         &  !< parameter 114 - [-] window transmissivity (not visual transmissivity) roof
       37.0_wp,        &  !< parameter 115 - [-] window albedo_type roof (albedo_type specified in radiation model)
       0.86_wp,        &  !< parameter 116 - [-] green emissivity roof
       5.0_wp,         &  !< parameter 117 - [-] green albedo_type roof (albedo_type specified in radiation model)
       0.0_wp,         &  !< parameter 118 - [-] green type roof
       0.75_wp,        &  !< parameter 119 - [-] shading factor
       0.70_wp,         &  !< parameter 120 - [-] g-value windows
       1.7_wp,         &  !< parameter 121 - [W/(m2*K)] u-value windows
       0.5_wp,         &  !< parameter 122 - [1/h] basic airflow without occupancy of the room for - summer 0.5_wp, winter 0.5
       2.0_wp,         &  !< parameter 123 - [1/h] additional airflow dependent on occupancy of the room for - summer 1.5_wp, winter 0.0
       0.6_wp,         &  !< parameter 124 - [-] heat recovery efficiency
       2.5_wp,         &  !< parameter 125 - [m2/m2] dynamic parameter specific effective surface
       165000.0_wp,    &  !< parameter 126 - [J/(m2*K)] dynamic parameter innner heat storage
       4.5_wp,         &  !< parameter 127 - [m2/m2] ratio internal surface/floor area
       0.0_wp,       &  !< parameter 128 - [W] maximal heating capacity
       -80.0_wp,         &  !< parameter 129 - [W] maximal cooling capacity
       5.0_wp,         &  !< parameter 130 - [W/m2] additional internal heat gains dependent on occupancy of the room
       2.1_wp,         &  !< parameter 131 - [W/m2] basic internal heat gains without occupancy of the room
       5.0_wp,         &  !< parameter 132 - [m] storey height
       0.2_wp,         &  !< parameter 133 - [m] ceiling construction height
       0.0_wp,         &  !< parameter 134 - [-] anthropogenic heat output for heating
       1.0_wp,       &  !< parameter 135 - [-] anthropogenic heat output for cooling
       1526000.0_wp,   &  !< parameter 136 - [J/(m3*K)] heat capacity 4th wall layer (inside) above ground floor level
       0.93_wp,         &  !< parameter 137 - [W/(m*K)] thermal conductivity 4th wall layer (inside) above ground floor level
       1526000.0_wp,   &  !< parameter 138 - [J/(m3*K)] capacity 4th wall layer (inside) ground floor level
       0.93_wp,         &  !< parameter 139 - [W/(m*K)] thermal conductivity 4th wall layer (inside) ground floor level
       709650.0_wp,    &  !< parameter 140 - [J/(m3*K)] heat capacity 4th wall layer (inside) ground plate
       0.12_wp,        &  !< parameter 141 - [W/(m*K)] thermal conductivity 4th wall layer (inside) ground plate
       1736000.0_wp,   &  !< parameter 142 - [J/(m3*K)] heat capacity 4th window layer (inside) ground floor level
       0.19_wp,        &  !< parameter 143 - [W/(m*K)] thermal conductivity 4th window layer (inside) ground floor level
       1736000.0_wp,   &  !< parameter 144 - [J/(m3*K)] heat capacity 4th layer (inside) above ground floor level
       0.45_wp,        &  !< parameter 145 - [W/(m*K)] thermal conductivity 4th window layer (inside) above ground floor level
       1526000.0_wp,   &  !< parameter 146 - [J/(m3*K)] heat capacity 4th wall layer (inside) roof
       0.93_wp,         &  !< parameter 147 - [W/(m*K)] thermal conductivity 4th wall layer (inside) roof
       1736000.0_wp,   &  !< parameter 148 - [J/(m3*K)] heat capacity 4th window layer (inside) roof
       0.19_wp         &  !< parameter 149 - [W/(m*K)] thermal conductivity 4th window layer (inside) roof
                        /)

 END SUBROUTINE usm_define_pars


 END MODULE urban_surface_mod
