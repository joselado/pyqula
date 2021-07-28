subroutine calculate_dos(values,nval,energies,ne,delta,weights,dos)
implicit none
integer, intent(in) :: nval
integer, intent(in) :: ne ! number of output energies
real (kind=8),intent(in) :: values(nval) ! input energies
real (kind=8),intent(in) :: weights(nval) ! weights
real (kind=8),intent(in) :: energies(ne) ! output energies
real (kind=8),intent(out) :: dos(ne) ! output dos
real (kind=8),intent(in) :: delta ! delta smearing
real (kind=8) :: delta2 ! delta smearing
real (kind=8) :: hold1(ne) ! holder
integer:: i ! counter 


dos(:) = 0.d00 ! initialize
delta2 = delta*delta


do i=1,nval ! sum over all the eigenvalues
  hold1 = energies - values(i) ! E - E_i
  hold1 = delta2 + hold1*hold1 ! delta2 + (E -E_i)**2
  hold1 = delta/hold1*weights(i) ! 1/.... times the weight
  dos = dos + hold1
enddo

return
end subroutine
