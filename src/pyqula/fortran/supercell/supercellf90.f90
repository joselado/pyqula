subroutine supercell2d(rin,numr,a1,a2,nc1,nc2,rout)
implicit none
integer, intent(in) :: numr ! number of input positions
integer, intent(in) :: nc1,nc2 ! size of the supercell
real (kind=8), intent(in) :: rin(numr,3) ! coordinates
real (kind=8), intent(in) :: a1(3),a2(3) ! vectors
real (kind=8), intent(out) :: rout(numr*nc1*nc2,3) ! output coordinates
integer :: i,j,k,l,m ! counters

l = 1 ! accumulator
do i=1,nc1
  do j=1,nc2
    do k=1,numr ! loop over atoms
      do m=1,3 ! loop over components
        rout(l,m) = rin(k,m) + a1(m)*i + a2(m)*j
      enddo
      l = l+1 ! increase counter
    enddo
  enddo
enddo


return
end subroutine

