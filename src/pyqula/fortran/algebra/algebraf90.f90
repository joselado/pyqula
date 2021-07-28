subroutine todouble(vin,vout,n,shift)
        implicit none
        integer, intent(in) :: n,shift
        complex (kind=8), intent(in) :: vin(n,n)
        complex (kind=8), intent(out) :: vout(n,2*n)
        integer :: i,j
        vout(:,:) = (0.d00,0.d00) ! output vector
        do i=1,n
          do j=1,n
            vout(i,2*j-1+shift) = vin(i,j)
          enddo
        enddo

        return
end subroutine
