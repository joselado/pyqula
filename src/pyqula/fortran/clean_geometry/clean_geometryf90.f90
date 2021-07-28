subroutine clean_geometry(rs,nr,minn,retain)
implicit none
integer, intent(in) :: nr,minn ! number of sites, minimum neighbors
real (kind=8), intent(in) :: rs(3,nr) ! sites
logical, intent(out) :: retain(nr) ! check whether to retain site i

integer :: i,j,acun ! counter
real (kind=8) :: dis,dr(3),ri(3) ! distance

do i=1,nr
  ri = rs(:,i)
  acun = 0 ! number of neighbors found
  do j=1,nr
    dr = ri - rs(:,j) ! distance in vector
    dis = dr(1)*dr(1) + dr(2)*dr(2) + dr(3)*dr(3) ! distance
    if ((dis.lt.1.1).and.(dis.gt.0.1)) acun = acun + 1 ! increase
  enddo
  ! check whether if enoght neighbors have been found
  retain(i) = acun.ge.minn ! retain
enddo

return

end subroutine
