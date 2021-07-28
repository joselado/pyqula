subroutine supermeanfield(vocc,nv,nsites,pairs,np,mout)
implicit none
integer, intent(in) :: nsites,nv ! number of sites
complex (kind=8), intent(in) :: vocc(nv,4*nsites) ! number of sites
integer, intent(in) :: np ! number of pair interactions
integer, intent(in) :: pairs(np,2) ! pairs that interact
complex (kind=8), intent(out) :: mout(4*nsites,4*nsites) ! output matrix
integer :: i,j,k,l,iw,ip ! counters
complex (kind=8) :: wf(4*nsites) ! number of sites

mout(:,:) = (0.d00,0.d00) ! output matrix


do iw=1,nv ! loop over wavefunctions
  wf(:) = vocc(iw,:) ! store
  do ip=1,np ! loop over pairs
    i = pairs(ip,1)
    j = pairs(ip,2)
    call addcontribution(mout,wf,4*i,4*j+3,.false.,nsites)
    call addcontribution(mout,wf,4*i,4*j+2,.true.,nsites)
    enddo ! close loop
  enddo ! close loop
enddo

mout = mout + transpose(conjg(mout)) ! Hermitian

return
end subroutine



! add contribution to the matrix
subroutine addcontribution(mout,wf,i,j,positive,nsites)
implicit none
integer, intent(in) :: nsites
complex (kind=8) :: mout(4*nsites,4*nsites) ! output matrix
integer :: i,j ! counters
logical :: positive
complex (kind=8) :: wf(4*nsites) ! number of sites

if (positive) then
  mout(i,j) = mout(i,j) + conjg(wf(j))*wf(i) ! add contribution
else
  mout(i,j) = mout(i,j) - conjg(wf(j))*wf(i) ! add contribution
endif


return
end subroutine

