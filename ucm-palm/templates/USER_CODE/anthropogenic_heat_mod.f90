!> @file anthropogenic_heat_mod.f90
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
! Copyright 2024 Singapore ETH Centre
! Copyright 2024 iMA Richter & Roeckle GmbH & Co. KG
! Copyright 1997-2024 Leibniz Universitaet Hannover
!--------------------------------------------------------------------------------------------------!
!
! Authors:
! ! --------
! @author Mathias Niffeler
! @author Tobias Gronemeier
!
!
! Description:
! ------------
!> This module introduces anthropogenic heat emissions from external sources into PALM. It considers
!> heat from buildings, traffic, and other point sources.
!> AH-profiles for BUILDINGS are typically obtained from a building energy model (BEM)
!> that replicates PALM's buildings in terms of geometry and materials, but applies a different
!> heating or cooling system to them.
!> POINT SOURCES are a more generic data format that can be used to represent any other source of
!> anthropogenic heat, e.g. incineration plants, power plants, or industrial complexes.
!>
!> Further work:
!> -------------
!> The module could be extended to include heat from traffic, if:
!>   a) a traffic model is available that provides heat profiles, and
!>   b) PALM4U started working with unique street IDs, identifying each street in the model domain.
!--------------------------------------------------------------------------------------------------!
 MODULE anthropogenic_heat_mod

#if defined( __parallel )
    USE MPI
#endif

    USE basic_constants_and_equations_mod,                                                         &
        ONLY:  pi

    USE control_parameters,                                                                        &
        ONLY:  coupling_char,                                                                      &
               external_anthropogenic_heat,                                                        &
               initializing_actions,                                                               &
               message_string,                                                                     &
               origin_date_time,                                                                   &
               time_since_reference_point

    USE cpulog,                                                                                    &
        ONLY:  cpu_log,                                                                            &
               log_point,                                                                          &
               log_point_s

    USE grid_variables,                                                                            &
        ONLY:  ddx,                                                                                &
               ddy,                                                                                &
               dx,                                                                                 &
               dy

    USE indices,                                                                                   &
        ONLY:  nx,                                                                                 &
               nxl,                                                                                &
               nxr,                                                                                &
               ny,                                                                                 &
               nyn,                                                                                &
               nys

    USE kinds

    USE netcdf_data_input_mod,                                                                     &
        ONLY:  char_fill,                                                                          &
               check_existence,                                                                    &
               close_input_file,                                                                   &
               get_attribute,                                                                      &
               get_dimension_length,                                                               &
               get_variable,                                                                       &
               init_model,                                                                         &
               input_file_ah,                                                                      &
               input_pids_ah,                                                                      &
               inquire_num_variables,                                                              &
               inquire_variable_names,                                                             &
               open_read_file

    USE pegrid

    USE surface_mod,                                                                               &
        ONLY:  surf_type,                                                                          &
               surf_def,                                                                           &
               surf_lsm,                                                                           &
               surf_usm

    IMPLICIT NONE

!
!-- Define data types
    TYPE int_dimension
       INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  val  !< 1d integer-value array
       LOGICAL ::  from_file = .FALSE.                  !< flag indicating whether an input variable is available and read from
                                                        !< file or default value is used
    END TYPE int_dimension

    TYPE real_2d_matrix
       REAL(wp) ::  fill                                !< fill value
       REAL(wp), DIMENSION(:,:), ALLOCATABLE ::  val    !< 2d real-value array
       LOGICAL ::  from_file = .FALSE.                  !< flag indicating whether an input variable is available and read from
                                                        !< file or default value is used
    END TYPE real_2d_matrix

    TYPE point_coordinates_t
       REAL(wp) ::  x_fill                              !< fill value for x-coordinates
       REAL(wp) ::  y_fill                              !< fill value for y-coordinates
       REAL(wp), DIMENSION(:), ALLOCATABLE ::  x        !< x-coordinates of point sources
       REAL(wp), DIMENSION(:), ALLOCATABLE ::  y        !< y-coordinates of point sources
       LOGICAL ::  from_file = .FALSE.                  !< flag indicating whether an input variable is available and read from
                                                        !< file or default value is used
    END TYPE point_coordinates_t

    TYPE type_building
       INTEGER(iwp) ::  id                                    !< building id
       INTEGER(iwp) ::  num_facades_h                         !< total number of horizontal facade elements

       INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  surf_inds  !< index array linking surface-element index with building
    END TYPE type_building

    TYPE surface_pointer
       TYPE(surf_type), POINTER :: surface_type => NULL()    !< pointer to a specific surface type
       INTEGER(iwp) :: surface_index                         !< index array linking surface-element index with building
    END TYPE surface_pointer

    TYPE point_to_surface_dictionary
       INTEGER(iwp) :: registered_point_sources = -1                     !< number of registered point sources
       INTEGER(iwp), DIMENSION(:), ALLOCATABLE :: point_source_ids       !< list of point source ids
       TYPE(surface_pointer), DIMENSION(:), ALLOCATABLE :: surface_list  !< list containing surface elements per building
    END TYPE point_to_surface_dictionary

!
!-- Define data structure for buidlings.
    TYPE build

       !> @todo add other parameters like ah emission profiles (if possible)

       INTEGER(iwp) ::  id                                !< building ID
       INTEGER(iwp) ::  kb_max                            !< highest vertical index of a building
       INTEGER(iwp) ::  kb_min                            !< lowest vertical index of a building
       INTEGER(iwp) ::  num_facades_per_building = 0      !< total number of horizontal and vertical facade elements
       INTEGER(iwp) ::  num_facades_per_building_h = 0    !< total number of horizontal facades elements
       INTEGER(iwp) ::  num_facades_per_building_l = 0    !< number of horizontal and vertical facade elements on local subdomain
       INTEGER(iwp) ::  num_facades_per_building_r = 0    !< total number of roof elements (horizontal upward facing elements)
       INTEGER(iwp) ::  num_facades_per_building_v = 0    !< total number of vertical facades elements

       INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  m              !< index array linking surface-element index with building
       INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  num_facade_h   !< number of horizontal facade elements per buidling
                                                                  !< and height level
       INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  num_facade_r   !< number of roof elements per buidling
                                                                  !< and height level
       INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  num_facade_v   !< number of vertical facades elements per buidling
                                                                  !< and height level


       LOGICAL ::  on_pe = .FALSE.   !< flag indicating whether a building with certain ID is on local subdomain

       REAL(wp) ::  area_facade           !< [m2] area of total facade
       REAL(wp) ::  building_height       !< building height
       REAL(wp) ::  vol_tot               !< [m3] total building volume
    END TYPE build

    TYPE(build), DIMENSION(:), ALLOCATABLE ::  buildings_ah  !< building array   !> @todo temporarily named buildings_ah to not confuse with var from indoor_model. --> rename


    REAL(wp), DIMENSION(:), ALLOCATABLE :: ah_time      !< time steps

    TYPE(int_dimension) :: building_ids                 !< ids of buildings with anthropogenic heat profiles
    TYPE(int_dimension) :: point_ids                    !< ids of point sources with anthropogenic heat profiles
    TYPE(int_dimension) :: street_ids                   !< ids of streets with anthropogenic heat profiles

    TYPE(point_coordinates_t) :: point_coords           !< exact coordinates of point sources

    TYPE(real_2d_matrix) :: building_ah                 !< anthropogenic heat profiles for buildings
    TYPE(real_2d_matrix) :: street_ah                   !< anthropogenic heat profiles for streets
    TYPE(real_2d_matrix) :: point_ah                    !< anthropogenic heat profiles for point sources

    TYPE(type_building), DIMENSION(:), ALLOCATABLE ::  building_surfaces  !< list containing surface elements per building
    TYPE(point_to_surface_dictionary) :: point_source_surfaces            !< dictionary linking point sources to surface elements

!
!-- Interfaces of subroutines accessed from outside of this module
    INTERFACE ah_parin
       MODULE PROCEDURE ah_parin
    END INTERFACE ah_parin

    INTERFACE ah_check_parameters
       MODULE PROCEDURE ah_check_parameters
    END INTERFACE ah_check_parameters

    INTERFACE ah_init
       MODULE PROCEDURE ah_init
    END INTERFACE ah_init

    INTERFACE ah_init_checks
       MODULE PROCEDURE ah_init_checks
    END INTERFACE ah_init_checks

    INTERFACE ah_actions
       MODULE PROCEDURE ah_actions
    END INTERFACE ah_actions


    SAVE

    PRIVATE

!
!-- Public functions
    PUBLIC                                                                                         &
       ah_parin,                                                                                   &
       ah_init,                                                                                    &
       ah_init_checks,                                                                             &
       ah_actions,                                                                                 &
       ah_check_parameters

!
!-- Public parameters, constants and initial values
    ! PUBLIC                                                                                        &
    ! @todo remove if not needed

 CONTAINS


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Parin for &anthropogenic_heat_par for external anthropogenic heat sources model
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE ah_parin

    IMPLICIT NONE

    CHARACTER(LEN=100) ::  line  !< dummy string that contains the current line of the parameter file

    INTEGER(iwp) ::  io_status   !< status after reading the namelist file

    LOGICAL ::  switch_off_module = .FALSE.  !< local namelist parameter to switch off the module
                                             !< although the respective module namelist appears in
                                             !< the namelist file

    NAMELIST /anthropogenic_heat_parameters/  switch_off_module

!
!-- Move to the beginning of the namelist file and try to find and read the namelist.
    REWIND( 11 )
    READ( 11, anthropogenic_heat_parameters, IOSTAT=io_status )

!
!-- Action depending on the READ status
    IF ( io_status == 0 )  THEN
!
!--    anthropogenic_heat_parameters namelist was found and read correctly.
!--    Enable the external anthropogenic heat module.
       IF ( .NOT. switch_off_module ) external_anthropogenic_heat = .TRUE.

    ELSEIF ( io_status > 0 )  THEN
!
!--    anthropogenic_heat_parameters namelist was found but contained errors.
!--    Print an error message including the line that caused the problem.
       BACKSPACE( 11 )
       READ( 11 , '(A)' ) line
       CALL parin_fail_message( 'anthropogenic_heat_parameters', line )

    ENDIF

 END SUBROUTINE ah_parin


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Initialization of the anthropogenic heat model
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE ah_init

    USE indices,                                                                                   &
        ONLY:  nxl,                                                                                &
               nxr,                                                                                &
               nyn,                                                                                &
               nys,                                                                                &
               nzb,                                                                                &
               nzt,                                                                                &
               topo_flags

    USE netcdf_data_input_mod,                                                                     &
        ONLY:  building_id_f

    IMPLICIT NONE

    INTEGER(iwp) ::  n_buildings  !< number of buildings
    INTEGER(iwp) ::  n_points     !< number of point sources


    INTEGER(iwp) ::  i           !< running index along x-direction
    INTEGER(iwp) ::  j           !< running index along y-direction
    INTEGER(iwp) ::  k           !< running index along z-direction
    INTEGER(iwp) ::  m           !< running index surface elements
    INTEGER(iwp) ::  nb          !< building index

    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  k_max_l             !< highest vertical index of a building on subdomain
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  k_min_l             !< lowest vertical index of a building on subdomain
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  n_fa                !< counting array
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  num_facades_h       !< dummy array used for summing-up total number of
                                                                    !< horizontal facade elements
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  num_facades_r       !< dummy array used for summing-up total number of
                                                                    !< roof elements
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  num_facades_v       !< dummy array used for summing-up total number of
                                                                    !< vertical facade elements
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  receive_dum_h       !< dummy array used for MPI_ALLREDUCE
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  receive_dum_r       !< dummy array used for MPI_ALLREDUCE
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE ::  receive_dum_v       !< dummy array used for MPI_ALLREDUCE


    CALL ah_profiles_netcdf_data_input
!
!-- Allocate point source surfaces data structure
    n_points = SIZE( point_ids%val, DIM=1 )
    ALLOCATE( point_source_surfaces%point_source_ids(0:n_points-1) )
    ALLOCATE( point_source_surfaces%surface_list(0:n_points-1) )

    point_source_surfaces%point_source_ids(:) = -9999

!
!-- Allocate building-surfaces data structure
    n_buildings = SIZE( building_ids%val, DIM=1 )
!
!-- Allocate building-data structure array. Note, this is a global array and all buildings with
!-- anthropogenic heat emission on domain are known by each PE. Height-dependent arrays and
!-- surface-element lists, however, are only allocated on PEs where the respective building is
!-- present (in order to reduce memory demands).
    ALLOCATE( buildings_ah(0:n_buildings-1) )

!
!-- Store building IDs and check if building with certain ID is present on subdomain.
    DO  nb = 0, n_buildings-1
       buildings_ah(nb)%id = building_ids%val(nb)

       IF ( ANY( building_id_f%var(nys:nyn,nxl:nxr) == buildings_ah(nb)%id ) )                        &
          buildings_ah(nb)%on_pe = .TRUE.
    ENDDO

!
!-- Determine the maximum vertical dimension occupied by each building.
    ALLOCATE( k_min_l(0:n_buildings-1) )
    ALLOCATE( k_max_l(0:n_buildings-1) )
    k_min_l = nzt + 1
    k_max_l = 0

    DO  i = nxl, nxr
       DO  j = nys, nyn
          IF ( building_id_f%var(j,i) /= building_id_f%fill )  THEN
             IF ( MINVAL( ABS( buildings_ah(:)%id - building_id_f%var(j,i) ), DIM=1 ) == 0 )  THEN
                nb = MINLOC( ABS( buildings_ah(:)%id - building_id_f%var(j,i) ), DIM=1 ) - 1
                DO  k = nzb, nzt+1
!
!--                Check if grid point belongs to a building.
                   IF ( BTEST( topo_flags(k,j,i), 6 ) )  THEN
                      k_min_l(nb) = MIN( k_min_l(nb), k )
                      k_max_l(nb) = MAX( k_max_l(nb), k )
                   ENDIF

                ENDDO
             ENDIF
          ENDIF
       ENDDO
    ENDDO

#if defined( __parallel )
    CALL MPI_ALLREDUCE( MPI_IN_PLACE, k_min_l, n_buildings, MPI_INTEGER, MPI_MIN, comm2d, ierr )
    CALL MPI_ALLREDUCE( MPI_IN_PLACE, k_max_l, n_buildings, MPI_INTEGER, MPI_MAX, comm2d, ierr )
#endif
    buildings_ah(:)%kb_min = k_min_l(:)
    buildings_ah(:)%kb_max = k_max_l(:)

    DEALLOCATE( k_min_l )
    DEALLOCATE( k_max_l )

!
!-- Allocate arrays for number of facades per height level. Distinguish between horizontal and
!-- vertical facades.
    DO  nb = 0, n_buildings - 1
       IF ( buildings_ah(nb)%on_pe )  THEN
          ALLOCATE( buildings_ah(nb)%num_facade_h(buildings_ah(nb)%kb_min:buildings_ah(nb)%kb_max) )
          ALLOCATE( buildings_ah(nb)%num_facade_r(buildings_ah(nb)%kb_min:buildings_ah(nb)%kb_max) )
          ALLOCATE( buildings_ah(nb)%num_facade_v(buildings_ah(nb)%kb_min:buildings_ah(nb)%kb_max) )

          buildings_ah(nb)%num_facade_h = 0
          buildings_ah(nb)%num_facade_r = 0
          buildings_ah(nb)%num_facade_v = 0
       ENDIF
    ENDDO
!
!-- Determine number of facade elements per building on local subdomain.
    buildings_ah(:)%num_facades_per_building_l = 0
    DO  m = 1, surf_usm%ns
!
!--    For the current facade element determine corresponding building index.
!--    First, obtain i,j,k indices of the building. Please note the offset between facade/surface
!--    element and building location (for horizontal surface elements the horizontal offsets are
!--    zero).
       i = surf_usm%i(m) + surf_usm%ioff(m)
       j = surf_usm%j(m) + surf_usm%joff(m)
       k = surf_usm%k(m) + surf_usm%koff(m)
!
!--    Determine building index and check whether building is listed in ah emission and is on PE.
       IF ( MINVAL( ABS( buildings_ah(:)%id - building_id_f%var(j,i) ), DIM=1 ) == 0 )  THEN
          nb = MINLOC( ABS( buildings_ah(:)%id - building_id_f%var(j,i) ), DIM=1 ) - 1

          IF ( buildings_ah(nb)%on_pe )  THEN
!
!--          Horizontal facade elements (roofs, etc.)
             IF ( surf_usm%upward(m)  .OR.  surf_usm%downward(m) )  THEN
!
!--             Count number of facade elements at each height level.
                buildings_ah(nb)%num_facade_h(k) = buildings_ah(nb)%num_facade_h(k) + 1
!
!--             Moreover, sum up number of local facade elements per building.
                buildings_ah(nb)%num_facades_per_building_l = buildings_ah(nb)%num_facades_per_building_l + 1
!
!--             Count roof elements at each height level
                IF ( surf_usm%upward(m) )                                                          &
                   buildings_ah(nb)%num_facade_r(k) = buildings_ah(nb)%num_facade_r(k) + 1
!
!--          Vertical facades
             ELSE
                buildings_ah(nb)%num_facade_v(k) = buildings_ah(nb)%num_facade_v(k) + 1
                buildings_ah(nb)%num_facades_per_building_l = buildings_ah(nb)%num_facades_per_building_l + 1
             ENDIF
          ENDIF
       ENDIF
    ENDDO
!
!-- Determine total number of facade elements per building and assign number to building data type.
    DO  nb = 0, n_buildings - 1
!
!--    Allocate dummy array used for summing-up facade elements.
!--    Please note, dummy arguments are necessary as building-date type arrays are not necessarily
!--    allocated on all PEs.
       ALLOCATE( num_facades_h(buildings_ah(nb)%kb_min:buildings_ah(nb)%kb_max) )
       ALLOCATE( num_facades_r(buildings_ah(nb)%kb_min:buildings_ah(nb)%kb_max) )
       ALLOCATE( num_facades_v(buildings_ah(nb)%kb_min:buildings_ah(nb)%kb_max) )
       ALLOCATE( receive_dum_h(buildings_ah(nb)%kb_min:buildings_ah(nb)%kb_max) )
       ALLOCATE( receive_dum_r(buildings_ah(nb)%kb_min:buildings_ah(nb)%kb_max) )
       ALLOCATE( receive_dum_v(buildings_ah(nb)%kb_min:buildings_ah(nb)%kb_max) )
       num_facades_h = 0
       num_facades_r = 0
       num_facades_v = 0
       receive_dum_h = 0
       receive_dum_r = 0
       receive_dum_v = 0

       IF ( buildings_ah(nb)%on_pe )  THEN
          num_facades_h = buildings_ah(nb)%num_facade_h
          num_facades_r = buildings_ah(nb)%num_facade_r
          num_facades_v = buildings_ah(nb)%num_facade_v
       ENDIF

#if defined( __parallel )
       CALL MPI_ALLREDUCE( num_facades_h,                                                          &
                           receive_dum_h,                                                          &
                           buildings_ah(nb)%kb_max - buildings_ah(nb)%kb_min + 1,                  &
                           MPI_INTEGER,                                                            &
                           MPI_SUM,                                                                &
                           comm2d,                                                                 &
                           ierr )

       CALL MPI_ALLREDUCE( num_facades_r,                                                          &
                           receive_dum_r,                                                          &
                           buildings_ah(nb)%kb_max - buildings_ah(nb)%kb_min + 1,                  &
                           MPI_INTEGER,                                                            &
                           MPI_SUM,                                                                &
                           comm2d,                                                                 &
                           ierr )

       CALL MPI_ALLREDUCE( num_facades_v,                                                          &
                           receive_dum_v,                                                          &
                           buildings_ah(nb)%kb_max - buildings_ah(nb)%kb_min + 1,                  &
                           MPI_INTEGER,                                                            &
                           MPI_SUM,                                                                &
                           comm2d,                                                                 &
                           ierr )
       IF ( ALLOCATED( buildings_ah(nb)%num_facade_h ) )  buildings_ah(nb)%num_facade_h = receive_dum_h
       IF ( ALLOCATED( buildings_ah(nb)%num_facade_r ) )  buildings_ah(nb)%num_facade_r = receive_dum_r
       IF ( ALLOCATED( buildings_ah(nb)%num_facade_v ) )  buildings_ah(nb)%num_facade_v = receive_dum_v
#else
       buildings_ah(nb)%num_facade_h = num_facades_h
       buildings_ah(nb)%num_facade_r = num_facades_r
       buildings_ah(nb)%num_facade_v = num_facades_v
#endif

!
!--    Deallocate dummy arrays
       DEALLOCATE( num_facades_h )
       DEALLOCATE( num_facades_r )
       DEALLOCATE( num_facades_v )
       DEALLOCATE( receive_dum_h )
       DEALLOCATE( receive_dum_r )
       DEALLOCATE( receive_dum_v )
!
!--    Allocate index arrays which link facade elements with surface-data type.
!--    Please note, no height levels are considered here (information is stored in surface-data type
!--    itself).
       IF ( buildings_ah(nb)%on_pe )  THEN
!
!--       Determine number of facade elements per building.
          buildings_ah(nb)%num_facades_per_building_h = SUM( buildings_ah(nb)%num_facade_h )
          buildings_ah(nb)%num_facades_per_building_r = SUM( buildings_ah(nb)%num_facade_r )
          buildings_ah(nb)%num_facades_per_building_v = SUM( buildings_ah(nb)%num_facade_v )
          buildings_ah(nb)%num_facades_per_building   = buildings_ah(nb)%num_facades_per_building_h +    &
                                                     buildings_ah(nb)%num_facades_per_building_v
!
!--       Allocate array that links the building with the urban-type surfaces.
!--       Please note, the linking array is allocated over all facade elements, which is
!--       required in case a building is located at the subdomain boundaries, where the building and
!--       the corresponding surface elements are located on different subdomains.
          ALLOCATE( buildings_ah(nb)%m(1:buildings_ah(nb)%num_facades_per_building_l) )
       ENDIF

    ENDDO
!
!-- Link facade elements with surface data type.
!-- Allocate array for counting.
    ALLOCATE( n_fa(0:n_buildings-1) )
    n_fa = 1

    DO  m = 1, surf_usm%ns
       i = surf_usm%i(m) + surf_usm%ioff(m)
       j = surf_usm%j(m) + surf_usm%joff(m)

       IF ( MINVAL( ABS( buildings_ah(:)%id - building_id_f%var(j,i) ), DIM=1 ) == 0 )  THEN
          nb = MINLOC( ABS( buildings_ah(:)%id - building_id_f%var(j,i) ), DIM=1 ) - 1

          IF ( buildings_ah(nb)%on_pe )  THEN
             buildings_ah(nb)%m(n_fa(nb)) = m
             n_fa(nb) = n_fa(nb) + 1
          ENDIF
       ENDIF
    ENDDO

 END SUBROUTINE ah_init


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Checks done after the Initialization.
!
!> @todo This routine is currently mis-used for additional initialization required to be done after
!>       the indoor_model init. Remove dependency to indoor_model_mod and move content of this
!>       routine to ah_init.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE ah_init_checks

    INTEGER(iwp) ::  b  !< loop index


    ALLOCATE( building_surfaces(LBOUND(building_ids%val, DIM=1):UBOUND(building_ids%val, DIM=1)) )

    DO  b = LBOUND(building_ids%val, DIM=1), UBOUND(building_ids%val, DIM=1)

       building_surfaces(b)%id = building_ids%val(b)

       CALL ah_building_id_to_surfaces(building_ids%val(b), building_surfaces(b)%surf_inds, building_surfaces(b)%num_facades_h)

    ENDDO

    ! @todo Map point sources to surface elements

 END SUBROUTINE ah_init_checks


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Reads anthropogenic heat profiles emitted from building and ground surfaces from a NetCDF file.
!--------------------------------------------------------------------------------------------------!
    SUBROUTINE ah_profiles_netcdf_data_input

       IMPLICIT NONE

       INTEGER(iwp) ::  id_netcdf                                    !< NetCDF id of input file
       INTEGER(iwp) ::  num_vars                                     !< number of variables in input file
       CHARACTER(LEN=100), DIMENSION(:), ALLOCATABLE ::  var_names   !< variable names in static input file


       INTEGER(iwp) ::  n_buildings = 0           !< number of buildings (for array allocation)
       INTEGER(iwp) ::  n_streets = 0             !< number of streets (for array allocation)
       INTEGER(iwp) ::  n_points = 0              !< number of point sources (for array allocation)
       INTEGER(iwp) ::  n_timesteps = 0           !< number of time steps (for array allocation)

!-- If no anthropogenic heat input file is available, skip this routine
       IF ( .NOT. input_pids_ah )  RETURN
!
!-- Measure CPU time
       CALL cpu_log( log_point_s(82), 'NetCDF input', 'start' )  ! @todo Check if log_point needs to be changed
!
!-- Skip the following if no expternal anthropogenic heat profiles are to be considered in the calculations.
       IF ( .NOT. external_anthropogenic_heat )  RETURN

#if defined ( __netcdf )
!
!-- Open file in read-only mode
       CALL open_read_file( TRIM( input_file_ah ) // TRIM( coupling_char ) , id_netcdf )
!
!-- Inquire all variable names.
!-- This will be used to check whether an optional input variable exists or not.
       CALL inquire_num_variables( id_netcdf, num_vars )

       ALLOCATE( var_names(1:num_vars) )
       CALL inquire_variable_names( id_netcdf, var_names )
!
!-- Read building ids from file
       IF ( check_existence( var_names, 'building_id' ) )  THEN

          CALL get_dimension_length( id_netcdf, n_buildings, 'building_id' )

          building_ids%from_file = .TRUE.

          ALLOCATE( building_ids%val(0:n_buildings-1) )

          CALL get_variable( id_netcdf, 'building_id', building_ids%val )
       ELSE
          building_ids%from_file = .FALSE.
       ENDIF
!
!-- Read streed ids from file
       IF ( check_existence( var_names, 'street_id' ) )  THEN

          CALL get_dimension_length( id_netcdf, n_streets, 'street_id' )

          street_ids%from_file = .TRUE.

          ALLOCATE( street_ids%val(0:n_streets-1) )

          CALL get_variable( id_netcdf, 'street_id', street_ids%val )
       ELSE
          street_ids%from_file = .FALSE.
       ENDIF
!
!-- Read point ids from file
       IF ( check_existence( var_names, 'point_id' ) )  THEN

         CALL get_dimension_length( id_netcdf, n_points, 'point_id' )

         point_ids%from_file = .TRUE.

         ALLOCATE( point_ids%val(0:n_points-1) )

         CALL get_variable( id_netcdf, 'point_id', point_ids%val )

      ELSE
         point_ids%from_file = .FALSE.
      ENDIF
!
!-- Read timesteps from file
       IF ( check_existence( var_names, 'time' ) .AND.                       &
            ( building_ids%from_file .OR. street_ids%from_file .OR. point_ids%from_file ) )  THEN

          CALL get_dimension_length( id_netcdf, n_timesteps, 'time' )

          ALLOCATE( ah_time(0:n_timesteps-1) )

          CALL get_variable( id_netcdf, 'time', ah_time )
       ENDIF
!
!-- Read anthrpogenic heat profiles from buildings from file
       IF ( building_ids%from_file .AND. check_existence( var_names, 'building_ah') ) THEN
          building_ah%from_file = .TRUE.
          CALL get_attribute( id_netcdf, char_fill, building_ah%fill, .FALSE., 'building_ah', .FALSE. )

          ALLOCATE( building_ah%val(0:n_timesteps-1,0:n_buildings-1) )

          CALL get_variable( id_netcdf, 'building_ah', building_ah%val, 0, n_buildings-1, 0, n_timesteps-1 )
          CALL ah_check_input_profiles('buildings', building_ah)
       ELSE
          building_ah%from_file = .FALSE.
       ENDIF
!
!-- Read anthropogenic heat profiles from points from file
       IF ( point_ids%from_file .AND. check_existence( var_names, 'point_ah') ) THEN
          point_ah%from_file = .TRUE.
          CALL get_attribute( id_netcdf, char_fill, point_ah%fill, .FALSE., 'point_ah', .FALSE. )

          ALLOCATE( point_ah%val(0:n_timesteps-1,0:n_points-1) )

          CALL get_variable( id_netcdf, 'point_ah', point_ah%val, 0, n_points-1, 0, n_timesteps-1 )
          CALL ah_check_input_profiles('points', point_ah)
       ELSE
          point_ah%from_file = .FALSE.
       ENDIF

!
!-- Read coordinates of point sources from file
       IF ( point_ah%from_file .AND. check_existence( var_names, 'point_x') .AND. check_existence( var_names, 'point_y') ) THEN
         point_coords%from_file = .TRUE.
         CALL get_attribute( id_netcdf, char_fill, point_coords%x_fill, .FALSE., 'point_x', .FALSE. )
         CALL get_attribute( id_netcdf, char_fill, point_coords%y_fill, .FALSE., 'point_y', .FALSE. )

         ALLOCATE( point_coords%x(0:n_points-1) )
         ALLOCATE( point_coords%y(0:n_points-1) )

         CALL get_variable( id_netcdf, 'point_x', point_coords%x )
         CALL get_variable( id_netcdf, 'point_y', point_coords%y )

         CALL ah_check_point_source_locations(point_coords)
      ELSE
         point_coords%from_file = .FALSE.
      ENDIF

!
!-- Finally, close input file
       CALL close_input_file( id_netcdf )
#endif
!
!-- End of CPU measurement
       CALL cpu_log( log_point_s(82), 'NetCDF input', 'stop' )

    END SUBROUTINE ah_profiles_netcdf_data_input


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Check anthropogenic heat profiles from input for buildings and point sources
!--------------------------------------------------------------------------------------------------!
    SUBROUTINE ah_check_input_profiles(source, ah_profile)

       IMPLICIT NONE

       CHARACTER(LEN=*), INTENT(IN) :: source            !< source of the anthropogenic heat profile
       TYPE(real_2d_matrix), INTENT(IN) :: ah_profile     !< anthropogenic heat profile

       !-- Check if the anthropogenic heat profile contains only non-negative values
       IF ( ANY( ( ah_profile%val < 0.0_wp ) .AND. ( ah_profile%val > ah_profile%fill )) ) THEN

         message_string = 'Anthropogenic heat profiles from ' // source // ' must be non-negative.'
         CALL message( 'netcdf_data_input_anthro_heat_profiles', 'AH0001', 1, 2, 0, 6, 0 )

       !-- Check if all values of the input anthropogenic heat profile have been defined
       ELSE IF ( ANY( ah_profile%val == ah_profile%fill ) ) THEN

         message_string = 'Some timesteps in the anthropogenic heat profile from ' // source // ' are not fully defined.'
         CALL message( 'netcdf_data_input_anthro_heat_profiles', 'AH0002', 0, 1, 0, 6, 0 )

       ENDIF

    END SUBROUTINE ah_check_input_profiles


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Check anthropogenic heat profiles from input for point sources
!--------------------------------------------------------------------------------------------------!
    SUBROUTINE ah_check_point_source_locations(point_coordinates)

       IMPLICIT NONE

       TYPE(point_coordinates_t), INTENT(IN) :: point_coordinates     !< point source coordinates

       INTEGER(iwp) :: p                           !< auxiliary index
       INTEGER(iwp) :: i_grid, j_grid              !< domain grid indices of the point source

       !-- Check if any of the point source coordinates are not fully defined
       IF ( ANY( point_coordinates%x == point_coordinates%x_fill ) .OR. &
             ANY( point_coordinates%y == point_coordinates%y_fill ) ) THEN

          message_string = 'Some point sources lack fully defined coordinates.' // ACHAR(10) // &
                            'Please make sure every point source is associated to a valid x and y coordinate.'
          CALL message( 'netcdf_data_input_anthro_heat_profiles', 'AH0003', 1, 2, 0, 6, 0 )

       ENDIF

       DO  p = LBOUND(point_coordinates%x, DIM=1), UBOUND(point_coordinates%x, DIM=1)
          CALL metric_coords_to_grid_indices(point_coordinates%x(p), point_coordinates%y(p), i_grid, j_grid)
          IF ( i_grid < 0 .OR. i_grid > nx .OR. j_grid < 0 .OR. j_grid > ny ) THEN
             message_string = 'Some point sources are located outside the model domain.'
             CALL message( 'netcdf_data_input_anthro_heat_profiles', 'AH0004', 1, 2, 0, 6, 0 )
          ENDIF
       ENDDO

    END SUBROUTINE ah_check_point_source_locations


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Perform actions for the anthropogenic heat model
!--------------------------------------------------------------------------------------------------!
    SUBROUTINE ah_actions( location )

       IMPLICIT NONE

       CHARACTER(LEN=*) ::  location

       CALL cpu_log( log_point(24), 'ah_actions', 'start' )  ! @todo Check if log_point needs to be changed

       SELECT CASE ( location )

          CASE ( 'after_pressure_solver' )
             !-- Apply anthropogenic heat profiles to the corresponding surfaces
             CALL ah_apply_to_surfaces

          CASE DEFAULT
             CONTINUE

       END SELECT

       CALL cpu_log( log_point(24), 'ah_actions', 'stop' )

    END SUBROUTINE ah_actions



!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> This routine applies the imported anthropogenic heat profiles to the corresponding urban surfaces.
!--------------------------------------------------------------------------------------------------!
    SUBROUTINE ah_apply_to_surfaces

       IMPLICIT NONE

       INTEGER(iwp) :: t_step, b, m, p                            !< auxiliary indices
       INTEGER(iwp), DIMENSION(:), ALLOCATABLE :: b_surf_indexes  !< array of building surface indexes
       INTEGER(iwp) ::  num_facades_per_building_h                !< total number of horizontal surfaces (up- and downward) per building
       INTEGER(iwp) :: b_surf_index                               !< building surface index

       TYPE(surface_pointer) :: p_surface                         !< pointer to the surface of an anthropogenic heat point source

       REAL(wp)  :: t_exact                                       !< copy of time_since_reference_point

       ! -- Identify the time step to which the anthropogenic heat profiles will be applied
       t_exact = time_since_reference_point
       t_step = MINLOC( ABS( ah_time - MAX( t_exact, 0.0_wp ) ), DIM = 1 ) - 1

       ! @todo Go over this routine again and make sure it is implemented correctly
       !       (Can the time step really only be 1 ahead of the current time upon restart?)
       ! --
       ! -- Note, in case of restart runs, the time step for the boundary data may indicate a time in
       ! -- the future. This needs to be checked and corrected.
       IF ( TRIM( initializing_actions ) == 'read_restart_data'  .AND.                                &
          ah_time(t_step) > t_exact )  THEN
          t_step = t_step - 1
       ENDIF

       !
       ! Identify the relevant surfaces and apply the anthropogenic heat profiles to them
       ! -- for buildings
       DO  b = LBOUND(building_surfaces, DIM=1), UBOUND(building_surfaces, DIM=1)

          !-- Only do something if surfaces of building are found (i.e., building is loaced on local pe)
          IF ( ALLOCATED( building_surfaces(b)%surf_inds ) )  THEN
             !-- Distribute the anthropogenic heat profile of the building to the corresponding surfaces
             DO  m = LBOUND(building_surfaces(b)%surf_inds, DIM=1), UBOUND(building_surfaces(b)%surf_inds, DIM=1)
                b_surf_index = building_surfaces(b)%surf_inds(m)
                IF ( surf_usm%upward(b_surf_index)  .OR.  surf_usm%downward(b_surf_index) )  THEN
                   surf_usm%waste_heat(b_surf_index) = ( building_ah%val(t_step + 1, b) * ( t_exact - ah_time(t_step) ) +   &  ! @bug dimensions of val might be wrong!
                                                       building_ah%val(t_step, b) * ( ah_time(t_step + 1) - t_exact ) )   &
                                                       / ( ah_time(t_step + 1) - ah_time(t_step) )                        &
                                                       / REAL( building_surfaces(b)%num_facades_h, wp )                   &
                                                       / (dx * dy)
                ENDIF
             ENDDO
          ENDIF
       ENDDO

       ! -- for point sources
       DO p = LBOUND(point_ids%val, DIM=1), UBOUND(point_ids%val, DIM=1)
          CALL get_matching_surface(point_ids%val(p), p_surface)
          IF ( .NOT. p_surface%surface_index == -9999 )  THEN
             p_surface%surface_type%waste_heat(p_surface%surface_index) =                                                 &
                         ( point_ah%val(t_step + 1, p) * ( t_exact - ah_time(t_step) ) +                                  &
                           point_ah%val(t_step, p) * ( ah_time(t_step + 1) - t_exact ) )                                  &
                         / ( ah_time(t_step + 1) - ah_time(t_step) )                                                      &
                         / (dx * dy)
          ENDIF
       ENDDO


    END SUBROUTINE ah_apply_to_surfaces


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> This routine fetches the roof surface tiles based on building ids.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE ah_building_id_to_surfaces( building_id, surf_indexes, num_facades_per_building_h )

    IMPLICIT NONE

    INTEGER(iwp), INTENT(IN) ::  building_id                               !< building id

    INTEGER(iwp), INTENT(OUT) ::  num_facades_per_building_h               !< total number of horzontal surfaces per building
    INTEGER(iwp), DIMENSION(:), ALLOCATABLE, INTENT(OUT) ::  surf_indexes  !< surface index

    INTEGER(iwp) ::  b  !< auxiliary building index


    DO  b = LBOUND( buildings_ah, DIM=1 ), UBOUND( buildings_ah, DIM=1 )
       IF ( buildings_ah(b)%id == building_id  .AND.  buildings_ah(b)%on_pe )  THEN
          ALLOCATE( surf_indexes(LBOUND( buildings_ah(b)%m, DIM=1 ):UBOUND( buildings_ah(b)%m, DIM=1 )) )
          surf_indexes = buildings_ah(b)%m
          num_facades_per_building_h = buildings_ah(b)%num_facades_per_building_h
          EXIT
       ENDIF
    ENDDO

 END SUBROUTINE ah_building_id_to_surfaces


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Fetch the corresponing surface pointer for a given id of an anthropogenic heat point source
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE get_matching_surface( point_id, surface )

    IMPLICIT NONE

    INTEGER(iwp),          INTENT(IN)  ::  point_id  !< point id
    TYPE(surface_pointer), INTENT(OUT) ::  surface   !< surface pointer

    INTEGER(iwp)                       ::  i         !< index of surface in the surface dictionary

!
!-- Check if the point source is already associated with a surface and fetch the correct index
    IF ( .NOT. ANY( point_source_surfaces%point_source_ids == point_id ) ) THEN
       CALL ah_point_id_to_surfaces( point_id )
    ENDIF

    i = MINLOC( ABS( point_source_surfaces%point_source_ids - point_id ), DIM = 1 ) - 1
!
!-- Assign the corresponding surface pointer to the surface
    surface = point_source_surfaces%surface_list(i)

 END SUBROUTINE get_matching_surface


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> This routine fetches the ground surface tiles based on point-source ids
!
!> @todo call routine once during init and save values to reduce computing time and output of
!>       warning message during every timestep
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE ah_point_id_to_surfaces( point_id )

    IMPLICIT NONE

    INTEGER(iwp), INTENT(IN) ::  point_id     !< point id

    INTEGER(iwp) ::  i                        !< auxiliary surface index
    INTEGER(iwp) ::  is                       !< auxiliary surface index
    INTEGER(iwp) ::  j                        !< auxiliary surface index
    INTEGER(iwp) ::  js                       !< auxiliary surface index
    INTEGER(iwp) ::  m                        !< auxiliary surface index
    INTEGER(iwp) ::  p                        !< auxiliary point source index
    INTEGER(iwp) ::  np                       !< number of registered point sources

    LOGICAL ::  on_core                       !< flag to indicate if the point source is on the local processor
    LOGICAL ::  found                         !< flag to indicate if a match was found for the point source

    REAL(wp) ::  x_coord_abs                  !< absolute metric x-coordinate of the point source
    REAL(wp) ::  y_coord_abs                  !< absolute metric y-coordinate of the point source

    TYPE(surface_pointer) ::  surface         !< surface pointer
    TYPE(point_to_surface_dictionary) :: temp_point_to_surface_dict  !< temporary point to surface dictionary

    found = .FALSE.

    IF ( point_coords%from_file )  THEN
!
!--    Retrieve metric coordinates of point sources from file
       x_coord_abs = point_coords%x(point_id)
       y_coord_abs = point_coords%y(point_id)
!
!--    Transform metric coordinates to grid indices
       CALL  metric_coords_to_grid_indices( x_coord_abs, y_coord_abs, is, js )
!
!--    Check if point is on local processor
       IF ( is >= nxl  .AND. is <= nxr .AND. js >= nys .AND. js <= nyn ) THEN
          on_core = .TRUE.
       ELSE
          on_core = .FALSE.
          surface%surface_type => surf_def
          surface%surface_index = -9999
       ENDIF
!
!--    Find the surface index of the grid cell in which the point source is located.
!--    First, search within the land surfaces (lsm)
       IF ( on_core )  THEN
          DO  m = 1, surf_lsm%ns
             i = surf_lsm%i(m)
             j = surf_lsm%j(m)
             IF ( i == is  .AND.  j == js )  THEN
                surface%surface_type => surf_lsm
                surface%surface_index = m
                found = .TRUE.
                IF ( .NOT.  ALLOCATED( surf_lsm%waste_heat ) )  THEN
                   ALLOCATE( surf_lsm%waste_heat(1:surf_lsm%ns) )
                   surf_lsm%waste_heat(:) = 0.0_wp
                ENDIF
                EXIT
             ENDIF
          ENDDO
       ENDIF
!
!--    If no match was found in the land surfaces, search the urban surfaces (usm).
       IF ( on_core .AND. .NOT. found )  THEN
          DO  m = 1, surf_usm%ns
             i = surf_usm%i(m)
             j = surf_usm%j(m)
             IF ( i == is  .AND.  j == js )  THEN
                surface%surface_type => surf_usm
                surface%surface_index = m
                found = .TRUE.
                IF ( .NOT.  ALLOCATED( surf_usm%waste_heat ) )  THEN
                   ALLOCATE( surf_usm%waste_heat(1:surf_usm%ns) )
                   surf_usm%waste_heat(:) = 0.0_wp
                ENDIF
                EXIT
             ENDIF
          ENDDO
       ENDIF
!
!--    If no match was found in the land or urban surfaces, search the default surfaces (def).
       IF ( on_core .AND. .NOT. found )  THEN
         DO  m = 1, surf_def%ns
            i = surf_def%i(m)
            j = surf_def%j(m)
            IF ( i == is  .AND.  j == js )  THEN
               surface%surface_type => surf_def
               surface%surface_index = m
               found = .TRUE.
               IF ( .NOT.  ALLOCATED( surf_def%waste_heat ) )  THEN
                  ALLOCATE( surf_def%waste_heat(1:surf_def%ns) )
                  surf_def%waste_heat(:) = 0.0_wp
               ENDIF
               EXIT
            ENDIF
         ENDDO
       ENDIF
!
!--    If a match was found add the surface to the point to surface dictionary
       IF ( on_core .AND. .NOT. found )  THEN
          surface%surface_type => surf_def
          surface%surface_index = -9999
          WRITE(message_string, '(A, I0, A)') 'The point source ', point_id, ' could not be attributed to any surface. Ignoring point source.'
          CALL message( 'ah_point_id_to_surfaces', 'AH0007', 0, 1, 0, 6, 0 )
       ENDIF
!
!--    Identify the number of registered point sources
       np = point_source_surfaces%registered_point_sources
!
!--    Register the newly identified point source
       point_source_surfaces%point_source_ids(np + 1) = point_id
       point_source_surfaces%surface_list(np + 1) = surface
!
!--    Update the number of registered point sources
       point_source_surfaces%registered_point_sources = np + 1

    ENDIF

 END SUBROUTINE ah_point_id_to_surfaces


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Transform metric point coordinates (x,y) to their equivalent grid point indices in the
!> model domain.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE metric_coords_to_grid_indices( x_coord_metric, y_coord_metric, i, j )

    IMPLICIT NONE

    REAL(wp), INTENT(IN) ::  x_coord_metric  !< x-coordinate of the point source
    REAL(wp), INTENT(IN) ::  y_coord_metric  !< y-coordinate of the point source

    INTEGER(iwp), INTENT(OUT) ::  i          !< x-index of the grid cell
    INTEGER(iwp), INTENT(OUT) ::  j          !< y-index of the grid cell

    REAL(wp) ::  x_coord_rel                 !< metric point x-coordinate relative to the model origin
    REAL(wp) ::  y_coord_rel                 !< metric point y-coordinate relative to the model origin
    REAL(wp) ::  x_coord                     !< point x-coordinate within the model coordinate system
    REAL(wp) ::  y_coord                     !< point y-coordinate within the model coordinate system

!
! -- offset metric point coordinates to reflect their relative position to the model origin
    x_coord_rel = x_coord_metric - init_model%origin_x
    y_coord_rel = y_coord_metric - init_model%origin_y

!
! -- Rotate the metric point coordinates to align with the model grid
    x_coord = COS( init_model%rotation_angle * pi / 180.0_wp ) * x_coord_rel                       &
            - SIN( init_model%rotation_angle * pi / 180.0_wp ) * y_coord_rel
    y_coord = SIN( init_model%rotation_angle * pi / 180.0_wp ) * x_coord_rel                       &
            + COS( init_model%rotation_angle * pi / 180.0_wp ) * y_coord_rel

!
! -- Then, compute the indices of the grid cell in which the point source is located.
    i = INT( x_coord * ddx, KIND = iwp )
    j = INT( y_coord * ddy, KIND = iwp )

 END SUBROUTINE metric_coords_to_grid_indices


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Check namelist parameter
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE ah_check_parameters

    USE control_parameters,                                                                        &
        ONLY: land_surface,                                                                        &
              urban_surface

!
!-- Check if the indoor module is activated
    IF ( .NOT. land_surface )  THEN

       message_string = 'Land-surface module is required when using anthropogenic heat module.'
       CALL message( 'ah_check_parameters', 'AH0005', 1, 2, 0, 6, 0 )

    ENDIF
!
!-- Check if the urban-surface module is activated
    IF ( .NOT. urban_surface )  THEN

       message_string = 'Urban-surface module is required when using anthropogenic heat module.'
       CALL message( 'ah_check_parameters', 'AH0006', 1, 2, 0, 6, 0 )

    ENDIF

 END SUBROUTINE ah_check_parameters


!--------------------------------------------------------------------------------------------------!
! Description:
! ------------
!> Print header for the anthropogenic heat model.
!> @todo Add information about activation to header file.
!--------------------------------------------------------------------------------------------------!
 SUBROUTINE ah_header

 END SUBROUTINE ah_header


 END MODULE anthropogenic_heat_mod
