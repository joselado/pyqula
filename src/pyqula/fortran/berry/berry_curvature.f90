subroutine berry_curvature(dhdx,dhdy,waves,es,n,op,delta,bout)
implicit none
integer, intent(in) :: n ! dimension
complex (kind=8), intent(in) :: dhdx(n,n),dhdy(n,n),waves(n,n),op(n,n)
real (kind=8), intent(in) :: es(n)
real (kind=8), intent(in) :: delta ! complex part to avoid blow up
real (kind=8), intent(out) :: bout ! berry curvature

complex (kind=8) :: opdhdx(n,n) ! product of two operators
integer :: i,j,ii,jj
complex (kind=8) :: hold1,hold2,fac

! initilice
bout = 0.d00

opdhdx = matmul(op,dhdx) + matmul(dhdx,op) ! product matrix
opdhdx = opdhdx/2.d00


do ii=1,n ! loop over states
  if (es(ii).gt.0.d00) cycle ! next iteration
  do jj=1,n ! loop over states
    if (ii==jj) cycle
    hold1 = 0.d00
    hold2 = 0.d00
    ! calculate matrix elements
    do j=1,n
      do i=1,n
        hold1 = hold1 + conjg(waves(ii,j))*dhdy(i,j)*waves(jj,i) ! add contribution
        hold2 = hold2 + conjg(waves(jj,j))*opdhdx(i,j)*waves(ii,i) ! add contribution
      enddo
    enddo
    ! add contribution to berry
    fac = es(ii)-es(jj) ! denominator
    fac = fac*fac + delta*delta ! denominator
    bout = bout + aimag(hold1*hold2/fac)
  enddo
enddo


end subroutine






subroutine berry_curvature_bands(dhdx,dhdy,waves,es,n,op,delta,bouts)
implicit none
integer, intent(in) :: n ! dimension
complex (kind=8), intent(in) :: dhdx(n,n),dhdy(n,n),waves(n,n),op(n,n)
real (kind=8), intent(in) :: es(n)
real (kind=8), intent(in) :: delta ! complex part to avoid blow up
real (kind=8), intent(out) :: bouts(n) ! berry curvature

complex (kind=8) :: opdhdx(n,n) ! product of two operators
integer :: i,j,ii,jj
complex (kind=8) :: hold1,hold2,fac

! initilice
bouts = 0.d00

opdhdx = matmul(op,dhdx) + matmul(dhdx,op) ! product matrix
opdhdx = opdhdx/2.d00


do ii=1,n ! loop over states
  do jj=1,n ! loop over states
    if (ii==jj) cycle
    hold1 = 0.d00
    hold2 = 0.d00
    ! calculate matrix elements
    do j=1,n
      do i=1,n
        hold1 = hold1 + conjg(waves(ii,j))*dhdy(i,j)*waves(jj,i) ! add contribution
        hold2 = hold2 + conjg(waves(jj,j))*opdhdx(i,j)*waves(ii,i) ! add contribution
      enddo
    enddo
    ! add contribution to berry
    fac = es(ii)-es(jj) ! denominator
    fac = fac*fac + delta*delta ! denominator
    bouts(ii) = bouts(ii) + aimag(hold1*hold2/fac)
  enddo
enddo


end subroutine















subroutine berry_curvature_bands_truncated(dhdx,dhdy,waves,es,n, &
                                  nw,op,delta,bouts)
implicit none
integer, intent(in) :: n ! dimension
integer, intent(in) :: nw ! number of waves used
complex (kind=8), intent(in) :: dhdx(n,n),dhdy(n,n),waves(nw,n),op(n,n)
real (kind=8), intent(in) :: es(nw)
real (kind=8), intent(in) :: delta ! complex part to avoid blow up
real (kind=8), intent(out) :: bouts(nw) ! berry curvature

complex (kind=8) :: opdhdx(n,n) ! product of two operators
integer :: i,j,ii,jj
complex (kind=8) :: hold1,hold2,fac

! initilice
bouts = 0.d00

opdhdx = matmul(op,dhdx) + matmul(dhdx,op) ! product matrix
opdhdx = opdhdx/2.d00


do ii=1,n ! loop over states
  do jj=1,n ! loop over states
    if (ii==jj) cycle
    hold1 = 0.d00
    hold2 = 0.d00
    ! calculate matrix elements
    do j=1,n
      do i=1,n
        hold1 = hold1 + conjg(waves(ii,j))*dhdy(i,j)*waves(jj,i) ! add contribution
        hold2 = hold2 + conjg(waves(jj,j))*opdhdx(i,j)*waves(ii,i) ! add contribution
      enddo
    enddo
    ! add contribution to berry
    fac = es(ii)-es(jj) ! denominator
    fac = fac*fac + delta*delta ! denominator
    bouts(ii) = bouts(ii) + aimag(hold1*hold2/fac)
  enddo
enddo

return

end subroutine


