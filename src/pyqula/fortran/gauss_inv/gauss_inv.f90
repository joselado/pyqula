! computes the desired element of the inverse in a tridiagonal matrix
! only for square blocks
subroutine gauss_inv(n,nm,ca,da,ua,i,j,g)
! number of blocks
integer, intent(in) :: nm
! size of the blocks
integer,intent(in) :: n
! element needed
integer, intent(in) :: i,j
! diagonal, uperdiagonal, subdiagonal
complex (kind=8), intent(in) :: ca(nm,n,n)
complex (kind=8), intent(in) :: ua(nm-1,n,n)
complex (kind=8), intent(in) :: da(nm-1,n,n)
! element needed
complex (kind=8), intent(out) :: g(n,n)
! arrays of the calculation
complex (kind=8) cl(n,n),cr(n,n),dl(n,n),dr(n,n)
complex (kind=8) a(n,n)
! holders
complex (kind=8) hm1(n,n),hm2(n,n)
! counters
integer i1,i2,i3,i4

cl=(0.d00,0.d00)
cr=(0.d00,0.d00)

dl=(0.d00,0.d00)
dr=(0.d00,0.d00)

do i1=1,n
cl(i1,i1)=(1.d00,0.d00) 
cr(i1,i1)=(1.d00,0.d00) 
enddo


! calculate dlii
do i1=1,i
  if (.not.(i1.eq.1)) then
  !  call selm(nm-1,n,da,a,i1-1)
    a(:,:) = da(i1-1,:,:)
    call multiply(a,dl,hm1,n,n,n)
  !  call selm(nm-1,n,ua,a,i1-1)
    a(:,:) = ua(i1-1,:,:)
    call multiply(hm1,a,dl,n,n,n)
  endif
  !call selm(nm,n,ca,a,i1)
  a(:,:) = ca(i1,:,:)
  dl=a-dl
  call inverse(dl,n)
  if((i.gt.j).and.(i1.le.i-1).and.(i1.ge.j)) then
  !  call selm(nm-1,n,da,a,i1)
    a(:,:) = da(i1,:,:)
    call multiply(a,dl,hm1,n,n,n)
    call multiply(hm1,cl,hm2,n,n,n)
    cl=-hm2
  endif
enddo



do i4=i,nm
  i1=nm+i-i4
  if (.not.(i1.eq.nm)) then
  !  call selm(nm-1,n,ua,a,i1)
    a(:,:) = ua(i1,:,:)
    call multiply(a,dr,hm1,n,n,n)
  !  call selm(nm-1,n,da,a,i1)
    a(:,:) = da(i1,:,:)
    call multiply(hm1,a,dr,n,n,n)
  endif
  !call selm(nm,n,ca,a,i1)
  a(:,:) = ca(i1,:,:)
  dr=a-dr
  call inverse(dr,n)
    if((i.lt.j).and.(i1.ge.i+1).and.(i1.le.j)) then
  !  call selm(nm-1,n,ua,a,i1-1)
    a(:,:) = ua(i1-1,:,:)
    call multiply(a,dr,hm1,n,n,n)
    call multiply(hm1,cr,hm2,n,n,n)
    cr=-hm2
    endif
enddo

   
call inverse(dr,n)
call inverse(dl,n)




!call selm(nm,n,ca,a,i)
a(:,:) = ca(i,:,:)
hm1=-a+dl+dr
call inverse(hm1,n)
g=hm1
if (i.lt.j) then
call multiply(hm1,cr,g,n,n,n)
endif
if (j.lt.i) then
call multiply(hm1,cl,g,n,n,n)
endif



return
end subroutine



! testing subroutine
subroutine test_inv(n,nm,ca,da,ua,i,j,g)
! number of blocks
integer nm
! size of the blocks
integer n
integer i,j
complex (kind=8) ca(nm,n,n),ua(nm-1,n,n),da(nm-1,n,n)
complex (kind=8) g(n,n)
complex (kind=8) cl(n,n),cr(n,n),dl(n,n),dr(n,n)
complex (kind=8) a(n,n)
complex (kind=8) hm1(n,n),hm2(n,n)
integer i1,i2,i3,i4,i5
complex (kind=8) ta(nm*n,nm*n)

ta=(0.d00,0.d00)

do i1=1,nm
  do i2=1,n
   do i3=1,n
   i4=(i1-1)*n
   ta(i4+i2,i4+i3)=ca(i1,i2,i3)
   enddo
  enddo
enddo


do i1=1,nm-1
  do i2=1,n
   do i3=1,n
   i4=(i1-1)*n
   i5=i1*n
   ta(i5+i2,i4+i3)=da(i1,i2,i3)
   ta(i4+i2,i5+i3)=ua(i1,i2,i3)
   enddo
  enddo
enddo


call inverse(ta,n*nm)


do i1=1,n
do i2=1,n
i3=(i-1)*n
i4=(j-1)*n
g(i1,i2)=ta(i3+i1,i2+i4)

enddo
enddo

return
end subroutine    






! computes the inverse of the matrix
subroutine inverse(A,n)
implicit none
! dimension
integer :: n
! matrix on imput inverse on output
complex (kind=8) :: A(n,n)
integer :: ipiv(n),info,lwork
complex (kind=8) :: work(1)

call ZGETRF( n, n, A, n, ipiv, info )

lwork=-1

call ZGETRI( n, A, n, ipiv, work, lwork, info )

lwork=n

call inv_aux(A,n,lwork,ipiv)


return
end subroutine

! inverse
subroutine inv_aux(A,n,lwork,ipiv)
implicit none
! dimension
integer :: n
!c matrix on imput inverse on output
complex (kind=8) A(n,n)
integer :: ipiv(n),info,lwork
complex (kind=8) :: work(lwork)

call ZGETRI( n, A, n, ipiv, work, lwork, info )
if (info.ne.0) then
write(*,*) 'Error in inverse',info,n
endif

return
end subroutine




! multiplyiply matrices, assume same dimension
subroutine multiply(A,B,C,d)
implicit none
integer :: n,k,m
integer :: d
complex (kind=8) :: A(d,d),B(d,d),C(d,d)
complex (kind=8) :: alpha,beta


m = d
k = d
n = d

alpha=(1.d00,0.d00)
beta=(0.d00,0.d00)

call ZGEMM('n','n',m,n,k,alpha,A,m,B,k,beta,C,m)

! C is the output

return
end subroutine multiply

