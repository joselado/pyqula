! module with sparse classes and functions
module sparse
implicit none
  ! type for sparse matrices
  type spmatrix
    character (len=20) :: name ! name of the matrix
    integer :: n ! dimension of the matrix
    integer :: nv ! number of non vanishing entries
    integer, allocatable :: i(:),j(:) ! indexes of the matrices
    complex (kind=8), allocatable :: mij(:) ! value of the entries
  endtype spmatrix


contains
  subroutine sparse_vmw(v,w,mat,n,res)
  implicit none
  integer, intent(in) :: n ! dimension
  complex (kind=8), intent(in) :: v(n),w(n)  ! vectors
  type(spmatrix), intent(in) :: mat ! sparse matrix
  complex(kind=8), intent(out) :: res ! result of the dot product  

  integer :: i,j,inv,nv  ! working counters

  res = (0.d00,0.d00)  ! initialice to 0 the result
  nv = mat%nv  ! number of nonvanishing elements
!  write(*,*) 
  do inv=1,nv
    i = mat%i(inv) ! index i
    j = mat%j(inv) ! index j
    res = res + conjg(v(i))*mat%mij(inv)*w(j)
  enddo
!  write(*,*) sum(conjg(v)*v),res
  end subroutine sparse_vmw


  ! sparse identity matrix
  subroutine sparse_id(n,mat)
  implicit none
  integer, intent(in) :: n ! dimension
  type(spmatrix), intent(out) :: mat ! sparse matrix
  integer :: ii
  ! deallocate if allocated
  if (allocated(mat%i)) deallocate(mat%i)
  if (allocated(mat%j)) deallocate(mat%j)
  if (allocated(mat%mij)) deallocate(mat%mij)
  ! allocate variables of the class
  allocate(mat%i(n))
  allocate(mat%j(n))
  allocate(mat%mij(n))
  do ii=1,n
    mat%i(ii) = ii
    mat%j(ii) = ii
    mat%mij(ii) = (1.d00,0.d00)
  enddo  
  mat%nv = n ! number of non vanishing elements
  return
  end subroutine sparse_id


  ! sparse density matrix, assuming SP calculation
  subroutine sparse_local_rho(iatom,mat)
  implicit none
  integer, intent(in) :: iatom ! dimension
  type(spmatrix), intent(out) :: mat ! sparse matrix
  integer :: ii
  ! deallocate if allocated
  if (allocated(mat%i)) deallocate(mat%i)
  if (allocated(mat%j)) deallocate(mat%j)
  if (allocated(mat%mij)) deallocate(mat%mij)
  ! allocate variables of the class
  allocate(mat%i(2))
  allocate(mat%j(2))
  allocate(mat%mij(2))

  mat%i(1) = 2*iatom-1
  mat%j(1) = 2*iatom-1
  mat%mij(1) = (1.d00,0.d00)

  mat%i(2) = 2*iatom
  mat%j(2) = 2*iatom
  mat%mij(2) = (1.d00,0.d00)


  mat%nv = 2 ! number of non vanishing elements
  return
  end subroutine sparse_local_rho


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! sparse density matrix, assuming SP calculation
  subroutine sparse_local_sx(n,iatom,mat)
  implicit none
  integer, intent(in) :: n,iatom ! dimension
  type(spmatrix), intent(out) :: mat ! sparse matrix
  integer :: ii
  ! deallocate if allocated
  if (allocated(mat%i)) deallocate(mat%i)
  if (allocated(mat%j)) deallocate(mat%j)
  if (allocated(mat%mij)) deallocate(mat%mij)
  ! allocate variables of the class
  allocate(mat%i(2))
  allocate(mat%j(2))
  allocate(mat%mij(2))
  mat%i(1) = 2*iatom-1
  mat%j(1) = 2*iatom
  mat%mij(1) = (1.d00,0.d00)
  mat%i(2) = 2*iatom
  mat%j(2) = 2*iatom-1
  mat%mij(2) = (1.d00,0.d00)
  mat%nv = 2 ! number of non vanishing elements
  return
  end subroutine sparse_local_sx



  subroutine sparse_local_sp(iatom,mat)
  implicit none
  integer, intent(in) :: iatom ! dimension
  type(spmatrix), intent(out) :: mat ! sparse matrix
  integer :: ii
  ! deallocate if allocated
  if (allocated(mat%i)) deallocate(mat%i)
  if (allocated(mat%j)) deallocate(mat%j)
  if (allocated(mat%mij)) deallocate(mat%mij)
  ! allocate variables of the class
  allocate(mat%i(1))
  allocate(mat%j(1))
  allocate(mat%mij(1))
  mat%i(1) = 2*iatom
  mat%j(1) = 2*iatom-1
  mat%mij(1) = (1.d00,0.d00)
  mat%nv = 1 ! number of non vanishing elements
  return
  end subroutine sparse_local_sp




  subroutine sparse_local_sm(iatom,mat)
  implicit none
  integer, intent(in) :: iatom ! dimension
  type(spmatrix), intent(out) :: mat ! sparse matrix
  integer :: ii
  ! deallocate if allocated
  if (allocated(mat%i)) deallocate(mat%i)
  if (allocated(mat%j)) deallocate(mat%j)
  if (allocated(mat%mij)) deallocate(mat%mij)
  ! allocate variables of the class
  allocate(mat%i(1))
  allocate(mat%j(1))
  allocate(mat%mij(1))
  mat%i(1) = 2*iatom-1
  mat%j(1) = 2*iatom
  mat%mij(1) = (1.d00,0.d00)
  mat%nv = 1 ! number of non vanishing elements
  return
  end subroutine sparse_local_sm





  subroutine sparse_local_sy(n,iatom,mat)
  implicit none
  integer, intent(in) :: n,iatom ! dimension
  type(spmatrix), intent(out) :: mat ! sparse matrix
  integer :: ii
  ! deallocate if allocated
  if (allocated(mat%i)) deallocate(mat%i)
  if (allocated(mat%j)) deallocate(mat%j)
  if (allocated(mat%mij)) deallocate(mat%mij)
  ! allocate variables of the class
  allocate(mat%i(2))
  allocate(mat%j(2))
  allocate(mat%mij(2))
  mat%i(1) = 2*iatom-1
  mat%j(1) = 2*iatom
  mat%mij(1) = (0.d00,-1.d00)
  mat%i(2) = 2*iatom
  mat%j(2) = 2*iatom-1
  mat%mij(2) = (0.d00,1.d00)
  mat%nv = 2 ! number of non vanishing elements
  return
  end subroutine sparse_local_sy


  subroutine sparse_local_sz(iatom,mat)
  implicit none
  integer, intent(in) :: iatom ! dimension
  type(spmatrix), intent(out) :: mat ! sparse matrix
  integer :: ii
  ! deallocate if allocated
  if (allocated(mat%i)) deallocate(mat%i)
  if (allocated(mat%j)) deallocate(mat%j)
  if (allocated(mat%mij)) deallocate(mat%mij)
  ! allocate variables of the class
  allocate(mat%i(2))
  allocate(mat%j(2))
  allocate(mat%mij(2))
  mat%i(1) = 2*iatom-1
  mat%j(1) = 2*iatom-1
  mat%mij(1) = (5.d-01,0.d00)
  mat%i(2) = 2*iatom
  mat%j(2) = 2*iatom
  mat%mij(2) = (-5.d-01,0.d00)
  mat%nv = 2 ! number of non vanishing elements
  return
  end subroutine sparse_local_sz













  ! sparse density matrix, without spin polarization
  subroutine sparse_local_rho_nonsp(n,iatom,mat)
  implicit none
  integer, intent(in) :: n,iatom ! dimension
  type(spmatrix), intent(out) :: mat ! sparse matrix
  integer :: ii
  ! deallocate if allocated
  if (allocated(mat%i)) deallocate(mat%i)
  if (allocated(mat%j)) deallocate(mat%j)
  if (allocated(mat%mij)) deallocate(mat%mij)
  ! allocate variables of the class
  allocate(mat%i(1))
  allocate(mat%j(1))
  allocate(mat%mij(1))

  mat%i(1) = iatom
  mat%j(1) = iatom
  mat%mij(1) = (1.d00,0.d00)


  mat%nv = 1 ! number of non vanishing elements
  return
  end subroutine sparse_local_rho_nonsp





  ! sparse Sz
  subroutine sparse_sz(n,mat)
  implicit none
  integer, intent(in) :: n ! dimension
  type(spmatrix), intent(out) :: mat ! sparse matrix
  integer :: ii
  ! deallocate if allocated
  if (allocated(mat%i)) deallocate(mat%i)
  if (allocated(mat%j)) deallocate(mat%j)
  if (allocated(mat%mij)) deallocate(mat%mij)


  ! allocate variables of the class
  allocate(mat%i(n))
  allocate(mat%j(n))
  allocate(mat%mij(n))
  do ii=1,n
    mat%i(ii) = ii
    mat%j(ii) = ii
    mat%mij(ii) = (-1.d00,0.d00)**(ii+1.d00)
  enddo  
  mat%nv = n ! number of non vanishing elements
  mat%n = n ! number of non vanishing elements
  mat%name = '     $S_z$     '   ! name of the matrix
  return
  end subroutine sparse_sz

  subroutine sparse_sx(n,mat)
  implicit none
  integer, intent(in) :: n ! dimension
  type(spmatrix), intent(out) :: mat ! sparse matrix
  integer :: ii
  ! deallocate if allocated
  if (allocated(mat%i)) deallocate(mat%i)
  if (allocated(mat%j)) deallocate(mat%j)
  if (allocated(mat%mij)) deallocate(mat%mij)
  ! allocate variables of the class
  allocate(mat%i(n))
  allocate(mat%j(n))
  allocate(mat%mij(n))
  do ii=0,n/2-1
    mat%i(2*ii+1) = 2*ii+1
    mat%j(2*ii+1) = 2*ii+2
    mat%mij(2*ii+1) = (1.d00,0.d00)
    mat%i(2*ii+2) = 2*ii+2
    mat%j(2*ii+2) = 2*ii+1
    mat%mij(2*ii+2) = (1.d00,0.d00)
  enddo  
  mat%nv = n ! number of non vanishing elements
  mat%n = n ! number of non vanishing elements
  mat%name = '    $S_x$    '   ! name of the matrix
  return
  end subroutine sparse_sx

  subroutine sparse_sy(n,mat)
  implicit none
  integer, intent(in) :: n ! dimension
  type(spmatrix), intent(out) :: mat ! sparse matrix
  integer :: ii
  ! deallocate if allocated
  if (allocated(mat%i)) deallocate(mat%i)
  if (allocated(mat%j)) deallocate(mat%j)
  if (allocated(mat%mij)) deallocate(mat%mij)
  ! allocate variables of the class
  allocate(mat%i(n))
  allocate(mat%j(n))
  allocate(mat%mij(n))
  do ii=0,n/2-1
    mat%i(2*ii+1) = 2*ii+1
    mat%j(2*ii+1) = 2*ii+2
    mat%mij(2*ii+1) = (0.d00,-1.d00)
    mat%i(2*ii+2) = 2*ii+2
    mat%j(2*ii+2) = 2*ii+1
    mat%mij(2*ii+2) = (0.d00,1.d00)
  enddo  
  mat%nv = n ! number of non vanishing elements
  mat%n = n ! number of non vanishing elements
  mat%name = '    $S_y$    '   ! name of the matrix
  return
  end subroutine sparse_sy


  subroutine sparse_sp(n,mat)
  implicit none
  integer, intent(in) :: n ! dimension
  type(spmatrix), intent(out) :: mat ! sparse matrix
  integer :: ii
  ! deallocate if allocated
  if (allocated(mat%i)) deallocate(mat%i)
  if (allocated(mat%j)) deallocate(mat%j)
  if (allocated(mat%mij)) deallocate(mat%mij)
  ! allocate variables of the class
  allocate(mat%i(n/2))
  allocate(mat%j(n/2))
  allocate(mat%mij(n/2))
  do ii=0,(n/2)-1
    mat%i(ii+1) = 2*ii+1
    mat%j(ii+1) = 2*ii+2
    mat%mij(ii+1) = (1.d00,0.d00)
  enddo  
  mat%nv = n/2 ! number of non vanishing elements
  mat%n = n ! dimension
  return
  end subroutine sparse_sp


  subroutine sparse_sm(n,mat)
  implicit none
  integer, intent(in) :: n ! dimension
  type(spmatrix), intent(out) :: mat ! sparse matrix
  integer :: ii
  ! deallocate if allocated
  if (allocated(mat%i)) deallocate(mat%i)
  if (allocated(mat%j)) deallocate(mat%j)
  if (allocated(mat%mij)) deallocate(mat%mij)
  ! allocate variables of the class
  allocate(mat%i(n/2))
  allocate(mat%j(n/2))
  allocate(mat%mij(n/2))
  do ii=0,(n/2)-1
    mat%i(ii+1) = 2*ii+2
    mat%j(ii+1) = 2*ii+1
    mat%mij(ii+1) = (1.d00,0.d00)
  enddo  
  mat%nv = n/2 ! number of non vanishing elements
  mat%n = n ! number of non vanishing elements
  return
  end subroutine sparse_sm






! sparse matrix with the indexes
  subroutine sparse_index(n,mat)
  implicit none
  integer, intent(in) :: n ! dimension
  type(spmatrix), intent(out) :: mat ! sparse matrix
  integer :: ii
  ! deallocate if allocated
  if (allocated(mat%i)) deallocate(mat%i)
  if (allocated(mat%j)) deallocate(mat%j)
  if (allocated(mat%mij)) deallocate(mat%mij)
  ! allocate variables of the class
  allocate(mat%i(n))
  allocate(mat%j(n))
  allocate(mat%mij(n))
  do ii=1,n
    mat%i(ii) = ii
    mat%j(ii) = ii
    mat%mij(ii) = ii
  enddo  
  mat%nv = n ! number of non vanishing elements
  mat%n = n ! number of non vanishing elements
  return
  end subroutine sparse_index


















end module sparse
