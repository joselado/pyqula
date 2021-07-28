subroutine mean_field_collinear0d(norb,intra,nocc,mixing,mfin, &
                    hubbard,error,infoscf,mfout,fermi,energy)
implicit none
integer, intent(in) :: norb ! number of orbitals
real (kind=8),intent(in) :: intra(norb,norb) ! intracell matrix
integer,intent(in) :: nocc ! number of occupated states
real (kind=8),intent(in) :: mfin(2,norb) ! initial mean field
real (kind=8),intent(in) :: mixing ! mixing
real (kind=8),intent(out) :: mfout(2,norb) ! final mean field
real (kind=8),intent(out) :: fermi ! initial mean field
real (kind=8),intent(in) :: hubbard
real (kind=8),intent(in) :: error
logical,intent(in) :: infoscf
real (kind=8),intent(out) :: energy ! total energy
! internal
real (kind=8) :: eup(norb),edn(norb),eud(2*norb) ! eigenvalues
real (kind=4) :: eud4(2*norb) ! eigenvalues
real (kind=8) :: intraup(norb,norb) ! intracell matrix
real (kind=8) :: intradn(norb,norb) ! intracell matrix
real (kind=8) :: denup(norb),dendn(norb),denup_old(norb),dendn_old(norb)
real (kind=8) :: vup(norb,norb),vdn(norb,norb) ! eigenvectors
real (kind=8) :: diff
real (kind=8) :: dnorm ! to normalize densities
integer :: i,info
! get initial densities
denup_old(:) = mfin(1,:)
dendn_old(:) = mfin(2,:)

do while (.true.)  ! infinite loop

  intraup = intra
  intradn = intra
!  write(*,*) sum(dendn_old+denup_old) 
  do i=1,norb ! create intra matrices
    intraup(i,i) = intra(i,i) + hubbard*dendn_old(i)
    intradn(i,i) = intra(i,i) + hubbard*denup_old(i)
  enddo
  
! diagonalize
  call diagonalize(eup,vup,intraup,norb)
  call diagonalize(edn,vdn,intradn,norb)

! get fermi energy
  do i=1,norb
    eud(i) = eup(i)
    eud(i+norb) = edn(i)
  enddo
  eud4=real(eud)
! sort the eigenvalues 
  call slasrt('I', 2*norb, eud4, info )
  fermi = dble(eud4(nocc)+eud4(nocc+1))/2.d00 ! number of occupied states
  ! initialize
  denup(:) = 0.d00
  dendn(:) = 0.d00
  energy = 0.d00 ! total energy
  do i=1,norb ! loop over eigenstates
    if (eup(i).lt.fermi) then
      denup(:) = denup(:) + vup(i,:)*vup(i,:) ! up contribution
      energy = energy + eup(i)  ! add energy
    endif
    if (edn(i).lt.fermi) then
      dendn(:) = dendn(:) + vdn(i,:)*vdn(i,:) ! down contribution
      energy = energy + edn(i) ! add energy
    endif
  enddo
  
  ! difference between old and new
  diff = sum(abs(denup(:)-denup_old(:)))
  diff = diff + sum(abs(dendn(:)-dendn_old(:)))
  ! normalize
  diff = diff/norb
  if (infoscf) then
    write(*,*) "Error = ",diff
  endif
  ! check whether to stop  
  if (diff.lt.error) exit
  
  ! update densities
  denup_old(:) = (1.d00-mixing)*denup_old(:) + mixing*denup(:)
  dendn_old(:) = (1.d00-mixing)*dendn_old(:) + mixing*dendn(:)

  ! normalize
  dnorm = nocc/sum(denup_old+dendn_old)
  denup_old = denup_old*dnorm
  dendn_old = dendn_old*dnorm
enddo ! end infinite loop

! store densities
mfout(1,:) = denup(:)
mfout(2,:) = dendn(:)

! and substract double counting
energy = energy - sum(denup*dendn)  

return
end subroutine




subroutine diagonalize(evals,evec,matrix,ndim)
implicit none
! number of atoms
integer :: ndim
! eigenvalues
real (kind=8) :: evals(ndim)
! complex matrix to diagonalize
real (kind=8) matrix(ndim,ndim)
real (kind=8) work(ndim*ndim)
! more variables...
integer :: info
real (kind=8) :: evec(ndim,ndim)

! all the eigenvalues
call DSYEV('V', 'U', ndim, matrix, ndim, evals, &
work,ndim*ndim ,info )


! output matrix is eigenvectors
    evec = transpose(matrix)  ! transpose it in order to be wf(1,:) an eigen
!    evec = matrix
    return
    end subroutine


