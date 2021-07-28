subroutine density_matrix(es,vs,smearing,ne,dimv,dm)
implicit none
integer, intent(in) :: dimv,ne ! dimensions
real (kind=8), intent(in) :: es(ne) ! vectors
complex (kind=8), intent(in) :: vs(ne,dimv) ! vectors
complex (kind=8), intent(out) :: dm(dimv,dimv) ! output matrix
complex (kind=8) :: v(dimv) ! vector
real (kind=8), intent(in) :: smearing ! smearing
real (kind=8) :: weight ! weight
integer :: i,j,k

dm(:,:) = (0.d00,0.d00) ! initialize


do k=1,ne
    weight = (1d00 - tanh(es(k)/smearing))/2d00 ! weight
    v(:) = vs(k,:) ! store
    do i=1,dimv
      do j=1,dimv
        dm(i,j) = dm(i,j) + conjg(v(i))*v(j)*weight
      enddo
    enddo
enddo


return
end subroutine
