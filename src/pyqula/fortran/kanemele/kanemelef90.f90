! return the Kane Mele vector between two vectors, given an intermediate
! set of vectors

subroutine kmvector(ri,rj,rms,nr,outv)
implicit none
integer, intent(in) :: nr
real (kind=8), intent(in) :: ri(3),rj(3),rms(nr,3)
real (kind=8), intent(out) :: outv(3) ! KM vector

integer i,j ! counters
real (kind=8) :: dr1(3),dr2(3)

outv(:) = 0.d00 ! initialize

do i=1,nr
  dr1(:) = rms(i,:) - ri(:)
  dr2(:) = -rms(i,:) + rj(:)
  if ((sum(dr1*dr1)<1.01).and.(sum(dr2*dr2)<1.01)) then
    outv(1) = dr1(2)*dr2(3) - dr1(3)*dr2(2)  
    outv(2) = dr1(3)*dr2(1) - dr1(1)*dr2(3)  
    outv(3) = dr1(1)*dr2(2) - dr1(2)*dr2(1)
    return ! return  
  endif
enddo



return
end subroutine
