
! return the pairs of points which are closer than d
subroutine number_neighborsf90(r1,r2,nr1,nr2,np)
implicit none
integer, intent(in) :: nr1,nr2 ! number of points
real (kind=8), intent(in) :: r1(3,nr1),r2(3,nr2)  ! points
integer, intent(out) :: np ! number of output points
integer :: i,j,ip ! counter
real (kind=8) :: dr,dx,dy,dz
np = 0
! use brute force
! check how many atoms
do i=1,nr1
  do j=1,nr2
    dx = r1(1,i)-r2(1,j)
    dy = r1(2,i)-r2(2,j)
    dz = r1(3,i)-r2(3,j)
    dr = dx*dx + dy*dy + dz*dz
    if ((0.9.lt.dr).and.(dr.lt.1.1)) then 
      np = np + 1 ! increase counter
    endif
  enddo
enddo

return
end subroutine
















! return the pairs of points which are closer than d
subroutine first_neighborsf90(r1,r2,pairs,nr1,nr2,np)
implicit none
integer, intent(in) :: nr1,nr2 ! number of points
real (kind=8), intent(in) :: r1(3,nr1),r2(3,nr2)  ! points
integer, intent(in) :: np ! number of output points
integer, intent(out) :: pairs(2,np)
integer :: i,j,ip ! counter
real (kind=8) :: dr,dx,dy,dz

pairs = 0

ip = 1 ! start counter

! do the loop again
do i=1,nr1
  do j=1,nr2
    dx = r1(1,i)-r2(1,j)
    dy = r1(2,i)-r2(2,j)
    dz = r1(3,i)-r2(3,j)
    dr = dx*dx + dy*dy + dz*dz
    if ((0.9.lt.dr).and.(dr.lt.1.1)) then 
      pairs(1,ip) = i - 1  ! store index
      pairs(2,ip) = j - 1 ! store index
      ip = ip + 1 ! increase counter
    endif
  enddo
enddo


return
end subroutine
