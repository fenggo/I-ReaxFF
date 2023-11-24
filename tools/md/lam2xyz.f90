!-------------------------------------------------------------------------
!   Convert Coordinate to XYZ format                                     -
!   Email: gfeng.alan@gmail.com                                          -
!-------------------------------------------------------------------------
!
program  lam2xyz
    implicit none

    real(8),allocatable::        x(:,:) 
    integer,allocatable::        types(:)
    character(2) ::              ctype(4) = (/'C','H','N','O'/)

    integer                      timestep,npart,n,nt,i,j,l
    integer                      nstep
    integer                      ina,inb,inc
    real(8)                      a(3),b(3),c(3)
    real(8)                      r1,r2,r3
    real(8)                      alpha,beta,gamma
    real(8)                      r_a,r_b,r_c
    real(8)                      delx,dely,delz
    real(8)                      cos_alpha,cos_beta,cos_gamma
    real(8)                      lo,hi
    real(8)                      xlo_bound, xhi_bound, xy, xlo, xhi
    real(8)                      ylo_bound, yhi_bound, xz, ylo, yhi
    real(8)                      zlo_bound, zhi_bound, yz, zlo, zhi 
    real(8) ::                   pi = 3.1415927
    character(len=40)            fin, fout
    character(80)                line
    logical            ::        apply_pbc=.true.

!    write(*,*)"input in file name:"
!    read(*,*)fin
!    write(*,*)"input out file name:"
!    read(*,*)fout
!    fin=trim(fin)
!    fout=trim(fout)
!    open(10,file=fin,status='old')
!    open(20,file=fout,status='unknown')

    read(5,*,end=200)
!
    read(5,*,end=200)
    read(5,*,end=200)
    read(5,*,end=200)npart
    read(5,'(A80)')line
!
    a = 0.0
    b = 0.0 
    c = 0.0 
!
    if(index(line,'ITEM: BOX BOUNDS')/=0)then
         if(index(line,'xy xz yz')/=0)then      
             read(5,*,end=200)xlo_bound, xhi_bound, xy
             read(5,*,end=200)ylo_bound, yhi_bound, xz
             read(5,*,end=200)zlo_bound, zhi_bound, yz
             xlo= xlo_bound - min(0.0,xy,xz,xy+xz)
             xhi= xhi_bound - MAX(0.0,xy,xz,xy+xz)
             ylo= ylo_bound - MIN(0.0,yz)
             yhi= yhi_bound - MAX(0.0,yz)
             zlo= zlo_bound 
             zhi= zhi_bound
             a = (/xhi-xlo, 0.0d0, 0.0d0/)
             b = (/xy, yhi-ylo, 0.0d0/)
             c = (/xz, yz, zhi-zlo/)

             r1 = a(1)*a(1)+ a(2)*a(2)+ a(3)*a(3)
             r1 = dsqrt(r1)
             r2 = b(1)*b(1)+ b(2)*b(2)+ b(3)*b(3)
             r2 = dsqrt(r2)
             r3 = c(1)*c(1)+ c(2)*c(2)+ c(3)*c(3)
             r3 = dsqrt(r3)

             cos_alpha = (b(1)*c(1) + b(2)*c(2) + b(3)*c(3))/(r2*r3)
             cos_beta  = (c(1)*a(1) + c(2)*a(2) + a(3)*c(3))/(r3*r1)
             cos_gamma = (a(1)*b(1) + a(2)*b(2) + a(3)*b(3))/(r1*r2)
         else
             read(5,*,end=200)lo,hi
             a(1)= hi-lo
             r1= a(1)
             read(5,*,end=200)lo,hi
             b(2)= hi-lo
             r2= b(2)
             read(5,*,end=200)lo,hi
             c(3)= hi-lo
             r3= c(3)

             cos_alpha = 0.0d0
             cos_beta  = 0.0d0
             cos_gamma = 0.0d0
         endif
    endif
!
    rewind(5)

    allocate(x(3,npart))
    allocate(types(npart))

    alpha = dacos(cos_alpha)*180.0d0/pi
    beta  = dacos(cos_beta)*180.0d0/pi
    gamma = dacos(cos_gamma)*180.0d0/pi

!    print '(3(A4,F12.4))','a=',r1,'b=',r2,'c=',r3
!    print '(A1)' , ' '

     !print '(A70)','***********************************************************************'!华丽的分割线
     !print '(2X,A40)','Cell parameters (Angstroms/Degrees):'
!     print '(A70)','***********************************************************************'
     !print '(A21)','Lattice vector:'
     !print '(3F12.6)',a(:)
     !print '(3F12.6)',b(:)
     !print '(3F12.6)',c(:)
     !print '(2(A8,F12.4))','a=',r1,'alpha=',alpha
     !print '(2(A8,F12.4))','b=',r2,'beta =', beta
     !print '(2(A8,F12.4))','c=',r3,'gamma=',gamma
     !print '(A70)','***********************************************************************'

    nstep = 0

    do   
       read(5,*,end=200)
       nstep= nstep + 1
       read(5,*,end=200)timestep
       read(5,*,end=200)
       read(5,*,end=200)npart
       write(6,300)npart
       read(5,'(A80)')line
!
!
       if(index(line,'ITEM: BOX BOUNDS')/=0)then
         if(index(line,'xy xz yz')/=0)then      
             read(5,*,end=200)xlo_bound, xhi_bound, xy
             read(5,*,end=200)ylo_bound, yhi_bound, xz
             read(5,*,end=200)zlo_bound, zhi_bound, yz
             xlo= xlo_bound - min(0.0,xy,xz,xy+xz)
             xhi= xhi_bound - MAX(0.0,xy,xz,xy+xz)
             ylo= ylo_bound - MIN(0.0,yz)
             yhi= yhi_bound - MAX(0.0,yz)
             zlo= zlo_bound 
             zhi= zhi_bound
             a = (/xhi-xlo, 0.0d0, 0.0d0/)
             b = (/xy, yhi-ylo, 0.0d0/)
             c = (/xz, yz, zhi-zlo/)

             r1 = a(1)*a(1)+ a(2)*a(2)+ a(3)*a(3)
             r1 = dsqrt(r1)
             r2 = b(1)*b(1)+ b(2)*b(2)+ b(3)*b(3)
             r2 = dsqrt(r2)
             r3 = c(1)*c(1)+ c(2)*c(2)+ c(3)*c(3)
             r3 = dsqrt(r3)

             cos_alpha = (b(1)*c(1) + b(2)*c(2) + b(3)*c(3))/(r2*r3)
             cos_beta  = (c(1)*a(1) + c(2)*a(2) + a(3)*c(3))/(r3*r1)
             cos_gamma = (a(1)*b(1) + a(2)*b(2) + a(3)*b(3))/(r1*r2)
         else
             read(5,*,end=200)lo,hi
             a(1)= hi-lo
             r1= a(1)
             read(5,*,end=200)lo,hi
             b(2)= hi-lo
             r2= b(2)
             read(5,*,end=200)lo,hi
             c(3)= hi-lo
             r3= c(3)

             cos_alpha = 0.0d0
             cos_beta  = 0.0d0
             cos_gamma = 0.0d0
         endif
       endif

       alpha = dacos(cos_alpha)*180.0d0/pi
       beta  = dacos(cos_beta)*180.0d0/pi
       gamma = dacos(cos_gamma)*180.0d0/pi

       write(6,500)timestep,r1,r2,r3,alpha,beta,gamma
       read(5,*,end=200)

       do i=1,npart 
          read(5,*,end=200)n,types(n),(x(j,n),j=1,3)
       enddo

       do i=1,npart
          if(apply_pbc) then
            r_c = (a(1)*b(2)-a(2)*b(1))*(b(2)*x(3,i)-b(3)*x(2,i)) - (a(3)*b(2)-a(2)*b(3))*(b(2)*x(1,i)-b(1)*x(2,i))
            r_c = r_c/((a(1)*b(2)-a(2)*b(1))*(c(3)*b(2)-c(2)*b(3)) - (a(3)*b(2)-a(2)*b(3))*(c(1)*b(2)-c(2)*b(1)))

            r_a = b(2)*x(1,i)-b(1)*x(2,i) - (c(1)*b(2)-c(2)*b(1))*r_c
            r_a = r_a/(a(1)*b(2)-a(2)*b(1))
            r_b = x(2,i)- r_a*a(2) - r_c*c(2)
            r_b = r_b/b(2)

            ina = anint(r_a)
            inb = anint(r_b)
            inc = anint(r_c)

            x(1,i)= x(1,i) - ina*a(1) - inb*b(1) - inc*c(1)
            x(2,i)= x(2,i) - ina*a(2) - inb*b(2) - inc*c(2)
            x(3,i)= x(3,i) - ina*a(3) - inb*b(3) - inc*c(3)
          endif
          write(6,400)ctype(types(i)),(x(j,i),j=1,3)
       enddo

    enddo
200 continue

300 FORMAT(I10)
400 format(A2,3F20.10)
500 format('Timestep:',I8,1X,'PBC:',6F10.5)

    deallocate(x)
    deallocate(types)

!    close(*)
!    close(*)

    stop
end
!
!
