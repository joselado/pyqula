! calculate the expansion in chevychev polynomials
subroutine  get_momentsf90(is,js,ms,v,numc,numij,numm,mus)
implicit none
! inputs
integer, intent(in) :: numc ! number of coefficients
integer, intent(in) :: numij ! number of entries of the matrix
integer, intent(in) :: numm ! dimension of the vector
integer, intent(in) :: is(numij),js(numij) ! indexes of the matrix
complex (kind=8), intent(in) :: ms(numij) ! entries of the matrix
complex (kind=8), intent(in) :: v(numm) ! input vector

! output
complex (kind=8), intent(out) :: mus(numc*2) ! coefficients of the expansion

! working varibles
complex (kind=8) :: a(numm)
complex (kind=8) :: am(numm)
complex (kind=8) :: ap(numm)
complex (kind=8) :: bk,bk1
complex (kind=8) :: mu0,mu1
integer i,j,k,l ! counter
! now go to work

! this routine is just the transcription in fortran of the
! routine in kpm.py


! first iteration
a(:) = 0.d00 ! initialize
am(:) = v(:) ! initialize
do k=1,numij  ! loop over non vanishing elements
  i = is(k) ! get index
  j = js(k) ! get index
  a(i) = a(i) + ms(k)*v(j)  ! perform product
enddo

bk = sum(conjg(v(:))*v(:)) ! scalar product
bk1 = sum(conjg(a(:))*v(:)) ! scalar product
! store first elements
mus(1) = bk
mus(2) = bk1

! now do the rest
do l=2,numc
  ap(:) = 0.d00 ! initialize
  do k=1,numij  ! loop over non vanishing elements
    i = is(k) ! get index
    j = js(k) ! get index
    ap(i) = ap(i) + 2.d00*ms(k)*a(j)  ! perform product
  enddo
  ap(:) = ap(:) - am(:) ! substract am, recursion relation
  bk = sum(conjg(a(:))*a(:)) ! scalar product
  bk1 = sum(conjg(ap(:))*a(:)) ! scalar product
  mus(2*l-1) = 2.d00*bk
  mus(2*l) = 2.d00*bk1
  am(:) = a(:) ! next iteration
  a(:) = ap(:) ! next iteration
enddo

! now substract first term
mu0 = mus(1)
mu1 = mus(2)
do l=2,numc
  mus(2*l-1) = mus(2*l-1) - mu0
  mus(2*l) = mus(2*l) - mu1
enddo



return
end subroutine get_momentsf90






subroutine  get_moments_ij(is,js,ms,numc,numij,numm,ii,jj,mus)
implicit none
! inputs
integer, intent(in) :: numc ! number of coefficients
integer, intent(in) :: numij ! number of entries of the matrix
integer, intent(in) :: numm ! dimension of the vector
integer, intent(in) :: is(numij),js(numij) ! indexes of the matrix
complex (kind=8), intent(in) :: ms(numij) ! entries of the matrix
integer, intent(in) :: ii,jj

! output
complex (kind=8), intent(out) :: mus(numc) ! coefficients of the expansion

! working varibles
complex (kind=8) :: v(numm) ! input vector
complex (kind=8) :: a(numm)
complex (kind=8) :: am(numm)
complex (kind=8) :: ap(numm)
complex (kind=8) :: bk,bk1
complex (kind=8) :: mu0,mu1
integer i,j,k,l ! counter
! now go to work

! this routine is just the transcription in fortran of the
! routine in kpm.py


v(:) = 0.d00
v(ii) = 1.d00 ! initialize

! first iteration
a(:) = 0.d00 ! initialize
am(:) = v(:) ! initialize
do k=1,numij  ! loop over non vanishing elements
  i = is(k) ! get index
  j = js(k) ! get index
  a(i) = a(i) + ms(k)*v(j)  ! perform product
enddo

bk = v(jj) ! scalar product
bk1 = a(jj) ! scalar product
! store first elements
mus(1) = bk
mus(2) = bk1

! now do the rest
do l=3,numc
  ap(:) = 0.d00 ! initialize
  do k=1,numij  ! loop over non vanishing elements
    i = is(k) ! get index
    j = js(k) ! get index
    ap(i) = ap(i) + 2.d00*ms(k)*a(j)  ! perform product
!    write(*,*) i,j
  enddo
  ap(:) = ap(:) - am(:) ! substract am, recursion relation
  bk = ap(jj) ! scalar product
  mus(l) = bk ! store
  am(:) = a(:) ! next iteration
  a(:) = ap(:) ! next iteration
!  write(*,*) l,ii,jj,numc
enddo



return
end subroutine get_moments_ij






! function to generate the ys
subroutine generate_profile(mus,xs,ys,dimmus,dimxs)
implicit none
integer, intent(in) :: dimmus,dimxs
complex (kind=8), intent(in) :: mus(dimmus),xs(dimxs)
complex (kind=8), intent(out) :: ys(dimxs)
complex (kind=8) :: tp(dimxs),t(dimxs),tm(dimxs)
integer :: i  ! counter

tm(:) = (1.d00,0.d00) ! initialize
t(:) = xs(:) ! initialize
ys(:) = mus(1)

! loop over moments
do i=2,dimmus
  ys = ys + 2.d00*mus(i)*t
  tp = 2.d00*xs*t - tm
  tm = t
  t = tp
enddo

ys = ys/sqrt(1.d00-xs*xs)

return
end subroutine generate_profile





! calculate the expansion in chevychev polynomials
subroutine  get_moments_vivj(is,js,ms,vi,vj,numc,numij,numm,mus)
implicit none
! inputs
integer, intent(in) :: numc ! number of coefficients
integer, intent(in) :: numij ! number of entries of the matrix
integer, intent(in) :: numm ! dimension of the vector
integer, intent(in) :: is(numij),js(numij) ! indexes of the matrix
complex (kind=8), intent(in) :: ms(numij) ! entries of the matrix
complex (kind=8), intent(in) :: vi(numm) ! first input vector
complex (kind=8), intent(in) :: vj(numm) ! second input vector

! output
complex (kind=8), intent(out) :: mus(numc*2) ! coefficients of the expansion

! working varibles
complex (kind=8) :: a(numm)
complex (kind=8) :: am(numm)
complex (kind=8) :: ap(numm)
complex (kind=8) :: bk,bk1
complex (kind=8) :: mu0,mu1
integer i,j,k,l ! counter
! now go to work

! this routine is just the transcription in fortran of the
! routine in kpm.py


! first iteration
a(:) = 0.d00 ! initialize
am(:) = vi(:) ! initialize
do k=1,numij  ! loop over non vanishing elements
  i = is(k) ! get index
  j = js(k) ! get index
  a(i) = a(i) + ms(k)*vi(j)  ! perform product
enddo

bk = sum(conjg(vj(:))*vi(:)) ! scalar product
bk1 = sum(conjg(vj(:))*a(:)) ! scalar product
! store first elements
mus(1) = bk
mus(2) = bk1

! now do the rest
do l=2,numc
  ap(:) = 0.d00 ! initialize
  do k=1,numij  ! loop over non vanishing elements
    i = is(k) ! get index
    j = js(k) ! get index
    ap(i) = ap(i) + 2.d00*ms(k)*a(j)  ! perform product
  enddo
  ap(:) = ap(:) - am(:) ! substract am, recursion relation
  bk = sum(conjg(vj(:))*ap(:)) ! scalar product
  mus(l) = bk
  am(:) = a(:) ! next iteration
  a(:) = ap(:) ! next iteration
enddo


return
end subroutine get_moments_vivj




