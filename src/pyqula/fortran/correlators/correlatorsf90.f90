subroutine correlators(vecs,pairs,numv,dimv,numc,corr)
implicit none
integer, intent(in) :: numv,dimv,numc  ! dimensions
complex (kind=8), intent(in) :: vecs(numv,dimv) ! input vectors
integer, intent(in) :: pairs(numc,2) ! pairs to calculate the correlators
complex (kind=8), intent(out) :: corr(numc) ! correlators
integer :: i,j,k1,k2 ! counters
complex (kind=8) :: tmp ! temporal storage


! write(*,*)  numv,dimv,numc

corr(:) = (0.d00,0.d00)

do i=1,numc ! loop over correlators
  tmp = (0.d00,0.d00) ! initialize
  k1 = pairs(i,1)+1 ! index
  k2 = pairs(i,2)+1 ! index
!  write(*,*) i,k1,k2
  do j=1,numv ! loop over vectors
!    write(*,*) tmp
    tmp = tmp + conjg(vecs(j,k2))*vecs(j,k1) ! add contribution
  enddo
  corr(i) = tmp ! store contribution
enddo


return
end subroutine






subroutine correlators_weighted(vecs,weight,pairs,numv,dimv,numc,corr)
implicit none
integer, intent(in) :: numv,dimv,numc  ! dimensions
complex (kind=8), intent(in) :: vecs(numv,dimv) ! input vectors
real (kind=8), intent(in) :: weight(numv) ! input vectors
integer, intent(in) :: pairs(numc,2) ! pairs to calculate the correlators
complex (kind=8), intent(out) :: corr(numc) ! correlators
integer :: i,j,k1,k2 ! counters
complex (kind=8) :: tmp ! temporal storage


! write(*,*)  numv,dimv,numc

corr(:) = (0.d00,0.d00)

do i=1,numc ! loop over correlators
  tmp = (0.d00,0.d00) ! initialize
  k1 = pairs(i,1)+1 ! index
  k2 = pairs(i,2)+1 ! index
!  write(*,*) i,k1,k2
  do j=1,numv ! loop over vectors
!    write(*,*) tmp
    tmp = tmp + conjg(vecs(j,k2))*vecs(j,k1)*weight(j) ! add contribution
  enddo
  corr(i) = tmp ! store contribution
enddo


!write(*,*) corr

return
end subroutine









subroutine multicorrelator(vecs,lambda,ijk,corr,dimv,numl,numc,numv)
implicit none
integer, intent(in) :: numv,dimv,numl,numc  ! dimensions
complex (kind=8), intent(in) :: vecs(numv,dimv) ! input vectors
complex (kind=8), intent(in) :: lambda(numl) ! coefficients
integer, intent(in) :: ijk(numl,3) ! pairs to calculate the correlators
complex (kind=8), intent(out) :: corr(numc) ! correlators
integer :: iv,i,j,k,l ! counters
complex (kind=8) :: tmp ! temporal storage

corr(:) = (0.d00,0.d00)

do l=1,numl ! loop over non zero elements
  tmp = (0.d00,0.d00) ! initialize
  i = ijk(l,1)+1 ! index
  j = ijk(l,2)+1 ! index
  k = ijk(l,3)+1 ! index
  tmp = (0.d00,0.d00) ! initialize
  do iv=1,numv ! loop over eigenvectors
    tmp = tmp + lambda(l)*conjg(vecs(iv,j))*vecs(iv,i)
  enddo
  corr(k) = corr(k) + tmp ! add
enddo


return
end subroutine



subroutine multicorrelator_bloch(vecs,ks,lambda,ijk,dir, &
                                         corr,dimv,numl,numc,numv)
implicit none
integer, intent(in) :: numv,dimv,numl,numc  ! dimensions
complex (kind=8), intent(in) :: vecs(numv,dimv) ! input vectors
real (kind=8), intent(in) :: ks(numv,3) ! input kpoints
complex (kind=8), intent(in) :: lambda(numl) ! coefficients
real (kind=8), intent(in) :: dir(numl,3) ! input direction
integer, intent(in) :: ijk(numl,3) ! pairs to calculate the correlators
complex (kind=8), intent(out) :: corr(numc) ! correlators
integer :: iv,i,j,k,l ! counters
complex (kind=8) :: tmp ! temporal storage
real (kind=8) :: pi
complex (kind=8) :: phi

corr(:) = (0.d00,0.d00)
pi = acos(-1.d00) ! pi


do l=1,numl ! loop over non zero elements
  tmp = (0.d00,0.d00) ! initialize
  i = ijk(l,1)+1 ! index
  j = ijk(l,2)+1 ! index
  k = ijk(l,3)+1 ! index
  tmp = (0.d00,0.d00) ! initialize
  do iv=1,numv ! loop over eigenvectors
    phi = dir(l,1)*ks(iv,1) + dir(l,2)*ks(iv,2) + dir(l,3)*ks(iv,3)
    phi = exp((0.d00,1.d00)*2.*pi*phi) ! complex phase
    tmp = tmp + lambda(l)*conjg(vecs(iv,j))*vecs(iv,i)*phi ! add contribution
  enddo
  corr(k) = corr(k) + tmp ! add
enddo


return
end subroutine


