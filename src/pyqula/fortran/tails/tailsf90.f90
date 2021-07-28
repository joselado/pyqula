subroutine tails(ws,nw,ns,ds)
implicit none
integer, intent(in) :: nw ! number of wavefunctions
integer, intent(in) :: ns ! number of sites
real (kind=8), intent(in) :: ws(nw,ns) ! wavefunctions
real (kind=8), intent(out) :: ds(nw,ns/2) ! output densities
real (kind=8) :: dtmp(ns/2) ! temporal density
real (kind=8) :: w2(ns) ! temporal storage
integer :: i,j,imax,ir,il ! various counters
integer :: ivec(1) ! various counters

ds(:,:) = 0.d00 ! initialize
do i=1,nw ! loop over wavefunctions
  w2(:) = ws(i,:)*ws(i,:) ! density
  imax = 1 ! initialize
  ivec = maxloc(w2) ! location of the maximum
  imax = ivec(1) ! maximum
  dtmp(:) = 0.d00 ! initialize
  do j=1,ns/2 ! loop over distances
    ir = modulo(imax+j-1,ns) ! distance to the right
    il = modulo(imax-j+1,ns) ! distance to the left
    if (ir.eq.0) ir = ns ! last one
    if (il.eq.0) il = ns ! last one
    dtmp(j) = w2(ir) + w2(il) ! density 
  enddo
  ds(i,:) = dtmp(:) ! store
enddo ! loop over waves




return
end subroutine tails
