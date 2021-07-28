! calculate the energy of a classical spin Hamiltonian

subroutine energy(thetas,phis,bs,js,indsjs,nspin,njs,eout)
implicit none
integer, intent(in) :: nspin ! number of spins
integer, intent(in) :: njs ! number of exchange interactions
integer, intent(in) :: indsjs(njs,2) ! indexes of interacting spins
real (kind=8), intent(in) :: js(njs,3,3) ! interaction matrix
real (kind=8), intent(in) :: thetas(nspin) ! theta values
real (kind=8), intent(in) :: phis(nspin) ! phi values
real (kind=8), intent(in) :: bs(nspin,3) ! magnetic fields
real (kind=8), intent(out) :: eout ! output energy

real (kind=8) :: mag(nspin,3) ! magnetic moment

integer :: i,s1,s2,j1,j2 ! counters

eout = 0.d00 ! initialize

mag(:,1) = sin(thetas)*cos(phis) ! Mx
mag(:,2) = sin(thetas)*sin(phis) ! My
mag(:,3) = cos(thetas) ! Mz


eout = sum(mag(:,1)*bs(:,1))
eout = eout + sum(mag(:,2)*bs(:,2))
eout = eout + sum(mag(:,3)*bs(:,3))

do i=1,njs ! loop over the different interactions
  s1 = indsjs(i,1) + 1 ! index of the first spin
  s2 = indsjs(i,2) + 1 ! index of the second spin
  do j1=1,3 ! loop over components
    do j2=1,3 ! loop over components
      eout = eout + mag(s1,j1)*js(i,j1,j2) *mag(s2,j2) ! add contribution 
    enddo
  enddo
enddo


return
end subroutine







subroutine jacobian(thetas,phis,bs,js,indsjs,nspin,njs,jac)
implicit none
integer, intent(in) :: nspin ! number of spins
integer, intent(in) :: njs ! number of exchange interactions
integer, intent(in) :: indsjs(njs,2) ! indexes of interacting spins
real (kind=8), intent(in) :: js(njs,3,3) ! interaction matrix
real (kind=8), intent(in) :: thetas(nspin) ! theta values
real (kind=8), intent(in) :: phis(nspin) ! phi values
real (kind=8), intent(in) :: bs(nspin,3) ! magnetic fields
real (kind=8), intent(out) :: jac(2*nspin) ! jacobian

real (kind=8) :: mag(nspin,3) ! magnetic moment
real (kind=8) :: dmdt(3),dmdp(3) ! derivatives of magnetic moment

integer :: i,s1,s2,j1,j2,k ! counters

jac(:) = 0.d00 ! initialize

mag(:,1) = sin(thetas)*cos(phis) ! Mx
mag(:,2) = sin(thetas)*sin(phis) ! My
mag(:,3) = cos(thetas) ! Mz

! part of the magnetic field
do i=1,nspin
  jac(i) = jac(i) + cos(thetas(i))*cos(phis(i))*bs(i,1)
  jac(i) = jac(i) + cos(thetas(i))*sin(phis(i))*bs(i,2)
  jac(i) = jac(i) - sin(thetas(i))*bs(i,3)
  jac(i+nspin) = jac(i+nspin) - sin(thetas(i))*sin(phis(i))*bs(i,1)
  jac(i+nspin) = jac(i+nspin) + sin(thetas(i))*cos(phis(i))*bs(i,2)
enddo

! still no part of magnetic field!!!!!

do i=1,njs ! loop over the different interactions
  s1 = indsjs(i,1) + 1 ! index of the first spin
  s2 = indsjs(i,2) + 1 ! index of the second spin
  ! derivative with respect to the first
  ! dM/dtheta
  dmdt(1) = cos(thetas(s1))*cos(phis(s1))   ! x component 
  dmdt(2) = cos(thetas(s1))*sin(phis(s1))   ! y component 
  dmdt(3) = -sin(thetas(s1))   ! z component 
  ! dM/dphi
  dmdp(1) = -sin(thetas(s1))*sin(phis(s1))   ! x component 
  dmdp(2) = sin(thetas(s1))*cos(phis(s1))   ! y component 
  dmdp(3) = 0.d00   ! z component 
  do j1=1,3 ! loop over components
    do j2=1,3 ! loop over components
      jac(s1) = jac(s1) + dmdt(j1)*js(i,j1,j2)*mag(s2,j2) 
      jac(nspin+s1) = jac(nspin+s1) + dmdp(j1)*js(i,j1,j2)*mag(s2,j2) 
    enddo
  enddo
  ! derivative with respect to the second
  ! dM/dtheta
  dmdt(1) = cos(thetas(s2))*cos(phis(s2))   ! x component 
  dmdt(2) = cos(thetas(s2))*sin(phis(s2))   ! y component 
  dmdt(3) = -sin(thetas(s2))   ! z component 
  ! dM/dphi
  dmdp(1) = -sin(thetas(s2))*sin(phis(s2))   ! x component 
  dmdp(2) = sin(thetas(s2))*cos(phis(s2))   ! y component 
  dmdp(3) = 0.d00   ! z component 
  do j1=1,3 ! loop over components
    do j2=1,3 ! loop over components
      jac(s2) = jac(s2) + dmdt(j2)*js(i,j1,j2)*mag(s1,j1) 
      jac(nspin+s2) = jac(nspin+s2) + dmdp(j2)*js(i,j1,j2)*mag(s1,j1) 
    enddo
  enddo
enddo


return
end subroutine
