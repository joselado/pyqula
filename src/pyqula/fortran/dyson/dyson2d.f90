subroutine dyson2d(intra,tx,ty,txy,txmy,n,nx,ny,nk,ez,g)
implicit none
! inputs
integer, intent(in) :: n ! simension of matrices
integer, intent(in) :: nx,ny ! replicas
complex (kind=8) , intent(in), dimension(n,n) :: intra,tx,ty,txy,txmy ! mat
complex (kind=8) , intent(in) :: ez ! complex energy
integer, intent(in) :: nk ! number of kpoints
! outputs
complex (kind=8) , intent(out), dimension(n*nx*ny,n*nx*ny) :: g ! green function
! internal
complex (kind=8) , dimension(2*nx+1,2*ny+1,n,n) :: gn ! vector of green
complex (kind=8) , dimension(nk*nk,n,n) :: gk ! inverse k green functions
complex (kind=8) , dimension(n,n) :: gt ! temporal green function
complex (kind=8) , dimension(n,n) :: ezid ! temporal green function
integer :: i,j,ik,i0,j0,i2,j2,ii,jj,ix1,ix2,iy1,iy2 ! counters
real (kind=8) :: pi,pi2 ! pi nmber
real (kind=8) :: kx,ky ! components of kpoints
real (kind=8), dimension(nk*nk,2) :: ks ! array with the different kpoints
complex (kind=8) :: im ! imaginary unit
integer :: icell ! counter for the replicas
integer :: indxy(nx*ny,2) ! indixes of that replica

pi = 3.141592 ! pi number
pi2 = 2.d00*pi
im = (0.d00,1.d00) ! imaginary unit

ik = 1 ! counter

! create the e*I
ezid(:,:) =  (0.d00,0.d00) ! initialize to zero
do i=1,n
  ezid(i,i) = ez ! diagonal part
enddo

! create all the 1/hk
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

do i=1,nk
  kx = dble(i)/dble(nk)
  do j=1,nk
    ky = dble(j)/dble(nk)
    ks(ik,1) = kx ! store kx component
    ks(ik,2) = ky ! store ky component
    gt = exp(im*kx*pi2)*tx + exp(im*ky*pi2)*ty + exp(im*(kx+ky)*pi2)*txy  + &
           exp(im*(kx-ky)*pi2)*txmy
    gt = gt + conjg(transpose(gt)) + intra ! full hamiltonian
    gt = ezid -gt ! redefine according to usual green function definition
    call inverse(gt,n) ! inverse function
    gk(ik,:,:) = gt(:,:) ! store green function
    ik = ik +1 ! increase kpoint counter
  enddo
enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! loop over replicas of the cell !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

icell = 1 ! initialize

do i=-nx,nx
  do j=-ny,ny
    gt(:,:) = (0.d00,0.d00) ! initialize to zero
    do ik=1,nk*nk ! loop over kpoints
      gt(:,:) = gt(:,:) +  &
         gk(ik,:,:)*exp(im*(ks(ik,1)*i+ks(ik,2)*j)*pi2) ! green times phase
    enddo
    gt = gt/(nk*nk) ! normalize
    gn(i+nx+1,j+ny+1,:,:) = gt(:,:) ! store in array
    icell = icell + 1 ! increase counter
  enddo
enddo

! get the indexes of the cells
icell = 1 ! initialize
do i=1,nx
  do j =1,ny
    indxy(icell,1) = i
    indxy(icell,2) = j
    icell = icell + 1 ! increase
  enddo
enddo



! now build up the big green function
do i=1,nx*ny ! loop over rows
  ix1 = indxy(i,1) ! index
  iy1 = indxy(i,2) ! index
  do j=1,nx*ny ! loop over columns
    ix2 = indxy(j,1) ! index
    iy2 = indxy(j,2) ! index
    ii = ix1-ix2 + nx + 1 ! get first index
    jj = iy1-iy2 + ny + 1 ! get second index
    gt(:,:) = gn(ii,jj,:,:) ! get green function
    i0 = n*(i-1) ! index offset
    j0 = n*(j-1) ! index offset
    do i2=1,n 
      do j2=1,n
        g(i0+i2,j0+j2) = gt(i2,j2) ! store element
      enddo 
    enddo
  enddo  ! close loop oever columns
enddo ! close loop over rows



return
end subroutine












! computes the inverse of the matrix
subroutine inverse(A,n)
implicit none
! dimension
integer :: n
! matrix on imput inverse on output
complex (kind=8) :: A(n,n)
integer :: ipiv(n),info,lwork
complex (kind=8) :: work(1)

call ZGETRF( n, n, A, n, ipiv, info )

lwork=-1

call ZGETRI( n, A, n, ipiv, work, lwork, info )

lwork=n

call inv_aux(A,n,lwork,ipiv)


return
end subroutine

! inverse
subroutine inv_aux(A,n,lwork,ipiv)
implicit none
! dimension
integer :: n
!c matrix on imput inverse on output
complex (kind=8) A(n,n)
integer :: ipiv(n),info,lwork
complex (kind=8) :: work(lwork)

call ZGETRI( n, A, n, ipiv, work, lwork, info )
if (info.ne.0) then
write(*,*) 'Error in inverse',info,n
endif

return
end subroutine





