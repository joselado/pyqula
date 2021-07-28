
!! norbitals is number of spinless orbitals
! i and j correspond to different spin flavours!!!
subroutine collinear_xychi(wf_chi_i,ene_chi_i,&
                           wf_chi_j,ene_chi_j, &
                           ene_chi, &
                           temperature_chi, &
                           delta, &
                           chi_total, &
                           norbitals, &
                           num_wf_i, &
                           num_wf_j, &
                           num_ene_chi)
  implicit none
  !!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! input variables
  !!!!!!!!!!!!!!!!!!!!!!!!!!!

  integer, intent(in) :: num_wf_i,num_wf_j ! dimensions of wavefunctions
  integer, intent(in) :: norbitals ! number of operators
  integer, intent(in) :: num_ene_chi ! number of energies
  complex (kind=8),intent(in) :: wf_chi_i(num_wf_i,norbitals)  ! WF i
  complex (kind=8),intent(in) :: wf_chi_j(num_wf_j,norbitals)  ! WF j
  real (kind=8),intent(in) :: ene_chi_i(num_wf_i)   ! Energy i
  real (kind=8),intent(in) :: ene_chi_j(num_wf_j)   ! Energy j
  real (kind=8),intent(in) :: ene_chi(num_ene_chi)   ! Energy for response
  real (kind=8),intent(in) :: temperature_chi   ! Energy for response
  real (kind=8),intent(in) :: delta   ! analytic cntinuation

  complex (kind=8),intent(out) :: chi_total(num_ene_chi,norbitals*norbitals)  ! WF i


  ! index for energy wave i and wave j
  integer :: ie,iwf1,iwf2
  real (kind=8) :: enetmp,etmp1,etmp2 ! temporal energies
  ! temporal wavefunctions
  complex (kind=8) :: wftmp1(norbitals),wftmp2(norbitals)
  complex (kind=8) :: chitmp  ! temporal chi
  complex (kind=8) :: ieps,im  ! smearing of Chi
  complex (kind=8) :: vawwbv,holdab  ! matrix element of the response
  real (kind=8) :: occ_fac,occ1,occ2 ! occupation factor 
  real (kind=8) :: den_res ! denominator of the response function
  integer :: iab,jab,kab ! counter for pair of operators

  im = (0.d00,1.d00)


  chi_total = 0.d00
  ieps = im*delta  ! complex infinitesimal

    do iwf1 = 1, num_wf_i ! first loop over states
      etmp1 = ene_chi_i(iwf1)
      wftmp1(:) = wf_chi_i(iwf1,:)
      do iwf2 = 1, num_wf_j  ! second loop over states
        etmp2 = ene_chi_j(iwf2)
        ! fermi energy has been put in 0
        call occupation(etmp1,temperature_chi,occ1)
        call occupation(etmp2,temperature_chi,occ2)
        occ_fac = occ2-occ1  ! occupation factor
        if (dabs(occ_fac).lt.1.d-04)  cycle  ! next iteration if too far
        ! if contribution is different from zero continue
        wftmp2(:) = wf_chi_j(iwf2,:)
! loop over the different matrices of linear response
        kab = 1
        do jab=1,norbitals   
        do iab=1,norbitals   
    ! calculate only at zero temperature
        ! calculate the matrix elementes <A><B>
          vawwbv = conjg(wftmp1(iab))*wftmp2(iab) ! first matrix element
          vawwbv = vawwbv * conjg(wftmp2(jab))*wftmp1(jab) ! second
  ! save in more accesible variables
          do ie = 1, num_ene_chi
            enetmp = ene_chi(ie)
            den_res = ((etmp1-etmp2) - enetmp)
            chitmp = occ_fac*vawwbv/(den_res+ieps)  ! add contribution
            ! ADD to the chi matrix array
            chi_total(ie,kab) = chi_total(ie,kab) + chitmp 
          enddo ! close loop over energies
        kab = kab +1 ! increase counter
        enddo ! close loop over AB
        enddo
    enddo  ! close loop over energies
  enddo ! close loop over the different matrices of linear response
  return
end subroutine collinear_xychi
! module with sparse classes and functions


  ! give the occupation at a given temperature
  subroutine occupation(energy,temperature,occ)
  implicit none
  real (kind=8), intent(out) :: occ
  real (kind=8), intent(in) :: temperature,energy
  ! if below threhold
  if (energy.lt.(-temperature)) then
  occ = 1.d00
  ! if above threhold
  else if (energy.gt.(temperature)) then
  occ = 0.d00
  else
  ! if in interval
  occ = 5.d-01 - (energy)/(2.d00*temperature)
  end if
  end subroutine occupation

