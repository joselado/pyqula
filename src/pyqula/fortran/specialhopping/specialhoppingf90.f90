subroutine twistedhopping(r1,r2,n,iout,jout,tout,nmax,nout, &
               cutoff,tinter,lamb,lambi,lambz,mint,dl)
implicit none
integer, intent(in) :: n ! number of sites
integer, intent(in) :: nmax ! maximum number of hoppings
real (kind=8), intent(in) :: r1(n,3),r2(n,3) ! positions
real (kind=8), intent(in) :: tinter ! interlayer hopping
real (kind=8), intent(in) :: cutoff,lambi,lamb,lambz,mint,dl ! parameters
real (kind=8), intent(out) :: tout(nmax) ! hoppings
integer, intent(out) :: iout(nmax),jout(nmax),nout ! indexes
integer :: i,j,k ! counter
real (kind=8) :: xi,yi,zi,xj,yj,zj,dx,dy,dz ! positions
real (kind=8) :: t,rij2,rij,cutoff2

cutoff2 = cutoff**2 ! square

k = 1 ! counter for the hoppings

do i=1,n ! loop over sites
  xi = r1(i,1)
  yi = r1(i,2)
  zi = r1(i,3)
  do j=1,n ! loop over sites
    if (i.eq.j) cycle
    xj = r2(j,1)
    yj = r2(j,2)
    zj = r2(j,3)
    dx = xi - xj
    dy = yi - yj
    dz = zi - zj
    rij2 = dx*dx + dy*dy + dz*dz ! distance**2
    if (rij2.gt.cutoff2) cycle ! too far, skip iteration
    if (rij2.lt.1.d-02) cycle ! same atom, skip iteration
    rij = sqrt(rij2) ! distance
    ! intralayer hopping
    t = -exp(-lamb*(rij - 1.d00))*(dx*dx+dy*dy)/rij2*exp(-lambz*dz*dz)
    t = t - tinter*exp(-lambi*(rij-dl))*dz*dz/rij2 ! interlayer hopping
    if (abs(t).lt.mint) cycle ! too small hopping
    if (k.gt.nmax) then
      write(*,*) "Not large enough array",nmax,k,mint,t
      return
    endif
    iout(k) = i ! store index
    jout(k) = j ! store index
    tout(k) = t ! store hopping
    k = k + 1 ! increase counter
  enddo
enddo

nout = k-1 ! number of hoppings 

return
end subroutine
