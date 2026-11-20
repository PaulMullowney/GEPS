! First-class ROCm PPM: same numerics as nvidia/ndslfv_pack_gpu.f90
! cyclic_cell_ppm_intp_two_loops_gpu, but work arrays are OTILE-wide.
! 1-rank TCo383 maps ~40GB hh+dqmono+qi; HIP totB is 96GiB with ~1.6GB free.

subroutine cyclic_cell_ppm_intp_two_loops_gpu(outer_index, outer_size, inner_size, &
                                              pp, pn, qq, lonn, nvars, sc, nstep_less)
   use const, only: RTYPE
   implicit none

   integer :: outer_index(3, outer_size), outer_size, inner_size, lonn, nvars
   real(kind=RTYPE), dimension(lonn + 1, inner_size, outer_size) :: pp, pn
   real(kind=RTYPE) :: qq(lonn, nvars, inner_size, outer_size)
   real(kind=RTYPE) :: sc, pn_t
   logical :: nstep_less(inner_size, outer_size)

   integer, parameter :: mono = 1
   integer, parameter :: otile = 128   ! 2026-09-19: 64 -> 128 (256 was no faster) (halves the launch count; prof11: NDSL is launch-gap bound); 2026-09-18: was 8 (OOM-era); 8 rows x 72 levels = 576 teams per launch, 48 tiles x 6 kernels per call -> 11 s/step of under-occupied kernels
   real(kind=RTYPE) hh(3*lonn, inner_size, otile)
   real(kind=RTYPE) qmi_t, qpi_t, qn_t
   real(kind=RTYPE) dqmono(3*lonn, nvars, inner_size, otile)
   real(kind=RTYPE) qi(3*lonn, nvars, inner_size, otile)
   integer kkh_array(lonn + 1, inner_size, otile)
   real(kind=RTYPE) tl_array(lonn + 1, inner_size, otile)
   real(kind=RTYPE) dql_array(lonn + 1, nvars, inner_size, otile)
   integer kstr(inner_size, otile), kend(inner_size, otile)
   real(kind=RTYPE) dqq_array(lonn, nvars, inner_size, otile)
   real(kind=RTYPE) locbndmin, locbndmax, pnlocs, locs1, locs2, locs3
   real(kind=RTYPE) dqi, dqimax, dqimin
   real(kind=RTYPE) tl, th, th2, th3, thp, thm, thc
   real(kind=RTYPE) :: dql_t, dqh_t
   real(kind=RTYPE) dpp, dqq, c1, c2, cc, rdthtl
   real(kind=RTYPE) rt, dqmono_pre, dqmono_cur, mass_pre, mass_t, mass_nxt, hh1, hh2
   integer i, j, kk, kkl, kkh, n, left, right, mid, kkn
   integer :: outer, inner, imp, imf, ot, og, ob, nloc
   integer :: async_id
   logical :: has_error, inside, accumulative

   ! locs_pp_map: define-side !$omp declare target in ndslfv_pack (acc2omp).
   ! Do not put declare target on this caller — that emits the whole host
   ! subroutine into the device LTO image (undefined geps_dbg_vram_ / wait).

   ! #region agent log
   integer(kind=8) :: dbg0, dbg1, dbg2
   interface
      subroutine geps_dbg_vram(hyp, locid, p0, p1, p2)
         integer hyp, locid
         integer(kind=8) p0, p1, p2
      end subroutine
   end interface
   ! #endregion

   async_id = 1
   has_error = .false.

   ! #region agent log
   dbg0 = int(lonn, 8); dbg1 = int(nvars, 8); dbg2 = int(outer_size, 8)
   call geps_dbg_vram(1, 500, dbg0, dbg1, dbg2)
   ! #endregion

   !$omp target enter data map(alloc:has_error)
   !$omp target update to(has_error)

   ! pn wrap does not need the fat work arrays; keep full outer.
   !$omp target teams distribute parallel do collapse(3) private(imp, imf, locbndmin, locbndmax, rt, pn_t)
   do outer = 1, outer_size
      do inner = 1, inner_size
         do i = 1, lonn + 1
            if (nstep_less(inner, outer)) then
               imp = outer_index(2, outer)
               imf = outer_index(3, outer)
               if (i .le. imf + 1) then
                  pn_t = pn(1, inner, outer)
                  call locs_pp_map(pp(1, inner, outer), imp + 4, lonn, imp, sc, locbndmin)
                  call locs_pp_map(pp(1, inner, outer), 2*imp - 4, lonn, imp, sc, locbndmax)
                  if (pn_t .lt. locbndmin - sc) then
                     rt = int((locbndmin - pn_t)/sc)*sc
                  else if (pn_t .gt. locbndmax) then
                     rt = (int((locbndmax - pn_t)/sc) - 1)*sc
                  else
                     rt = 0.0
                  end if
                  pn(i, inner, outer) = pn(i, inner, outer) + rt
               end if
            end if
         end do
      end do
   end do

   ! #region agent log
   dbg0 = int(lonn, 8); dbg1 = int(nvars, 8); dbg2 = int(otile, 8)
   call geps_dbg_vram(1, 663, dbg0, dbg1, dbg2)
   ! #endregion
   !$omp target enter data map(alloc:kstr, kend, kkh_array, hh, dqmono, qi, tl_array, dql_array, dqq_array)
   ! #region agent log
   dbg0 = int(lonn, 8); dbg1 = int(nvars, 8); dbg2 = int(otile, 8)
   call geps_dbg_vram(1, 664, dbg0, dbg1, dbg2)
   ! #endregion

   do ob = 1, outer_size, otile
      nloc = min(otile, outer_size - ob + 1)

      ! 2026-11-20: kstr/kend cell search parallel over i. The serial scans kept the
      ! last match: the smallest i for the downward scans, the largest for the
      ! upward ones, so atomic min/max over the same tests gives the same cells
      ! (no monotonicity of the cell edges assumed). One thread per (ot, inner)
      ! walked up to 2*imp cells: 22 ms/step on TCo383 8-GPU.
      !$omp target teams distribute parallel do collapse(2) private(og, imf) firstprivate(ob, nloc)
      do ot = 1, nloc
         do inner = 1, inner_size
            og = ob + ot - 1
            if (nstep_less(inner, og)) then
               imf = outer_index(3, og)
               kstr(inner, ot) = 0
               if (pn(1, inner, og) .lt. pp(1, inner, og)) kstr(inner, ot) = huge(1)
               kend(inner, ot) = 0
               if (pn(imf + 1, inner, og) .lt. pp(1, inner, og) + sc) kend(inner, ot) = huge(1)
            end if
         end do
      end do

      !$omp target teams distribute parallel do collapse(3) private(og, imp, imf, pn_t, pnlocs, locs1, locs2, kkn) firstprivate(ob, nloc)
      do ot = 1, nloc
         do inner = 1, inner_size
            do i = 1, 3*lonn - 1
               og = ob + ot - 1
               if (nstep_less(inner, og)) then
                  imp = outer_index(2, og)
                  if (i .le. 3*imp - 1) then
                     imf = outer_index(3, og)
                     call locs_pp_map(pp(1, inner, og), i, lonn, imp, sc, locs1)
                     call locs_pp_map(pp(1, inner, og), i + 1, lonn, imp, sc, locs2)
                     pn_t = pn(1, inner, og)
                     pnlocs = pp(1, inner, og)
                     if ((pn_t .ge. locs1) .and. (pn_t .lt. locs2)) then
                        if (pn_t .lt. pnlocs) then
                           if (i .le. imp) then
                              !$omp atomic update
                              kstr(inner, ot) = min(kstr(inner, ot), i)
                           end if
                        else if (i .ge. imp + 1 .and. i .le. 2*imp) then
                           !$omp atomic update
                           kstr(inner, ot) = max(kstr(inner, ot), i)
                        end if
                     end if
                     pn_t = pn(imf + 1, inner, og)
                     pnlocs = pp(1, inner, og) + sc
                     if (pn_t .ge. locs1 .and. pn_t .lt. locs2) then
                        kkn = i + 1
                        if (pn_t .lt. pnlocs) then
                           if (i .ge. imp .and. i .le. 2*imp) then
                              !$omp atomic update
                              kend(inner, ot) = min(kend(inner, ot), kkn)
                           end if
                        else if (i .ge. 2*imp + 1) then
                           !$omp atomic update
                           kend(inner, ot) = max(kend(inner, ot), kkn)
                        end if
                     end if
                  end if
               end if
            end do
         end do
      end do

      !$omp target teams distribute parallel do collapse(2) private(og, imp) firstprivate(ob, nloc)
      do ot = 1, nloc
         do inner = 1, inner_size
            og = ob + ot - 1
            if (nstep_less(inner, og)) then
               imp = outer_index(2, og)
               if (kstr(inner, ot) .eq. huge(1)) kstr(inner, ot) = 0
               if (kstr(inner, ot) .eq. 0) has_error = .true.
               kstr(inner, ot) = max(3, kstr(inner, ot))
               if (kend(inner, ot) .eq. huge(1)) kend(inner, ot) = 0
               if (kend(inner, ot) .eq. 0) has_error = .true.
               kend(inner, ot) = min(3*imp - 2, kend(inner, ot))
            end if
         end do
      end do

      ! 2026-09-18: kernels 2+3 fused and parallel over i (binary searches are
      ! independent); the range check that used to be a separate kernel is
      ! done on the freshly computed pair. Same arithmetic, same results.
      !$omp target teams distribute parallel do collapse(3) private(og, imp, imf, left, right, mid, locs1, kkl, kkh) firstprivate(ob, nloc)
      do ot = 1, nloc
         do j = 1, inner_size
            do i = 0, lonn
               og = ob + ot - 1
               if (nstep_less(j, og)) then
                  imp = outer_index(2, og)
                  imf = outer_index(3, og)
                  if (i .eq. 0) then
                     kkh_array(1, j, ot) = kstr(j, ot)
                  else if (i .le. imf) then
                     left = kstr(j, ot)
                     right = kend(j, ot) + 1
                     do while (right - left > 1)
                        mid = (left + right)/2
                        call locs_pp_map(pp(1, j, og), mid, lonn, imp, sc, locs1)
                        if (pn(i + 1, j, og) .lt. locs1) then
                           right = mid
                        else
                           left = mid
                        end if
                     end do
                     kkh_array(i + 1, j, ot) = left
                     kkh = left
                     if (kkh .eq. kend(j, ot) + 2) has_error = .true.
                  end if
               end if
            end do
         end do
      end do
      !$omp target teams distribute parallel do collapse(3) private(og, imf, kkl, kkh) firstprivate(ob, nloc)
      do ot = 1, nloc
         do inner = 1, inner_size
            do i = 1, lonn
               og = ob + ot - 1
               if (nstep_less(inner, og)) then
                  imf = outer_index(3, og)
                  if (i .le. imf) then
                     kkl = kkh_array(i, inner, ot)
                     kkh = kkh_array(i + 1, inner, ot)
                     if (kkh .lt. kkl) has_error = .true.
                  end if
               end if
            end do
         end do
      end do

      !$omp target teams distribute parallel do collapse(3) private(og, imp, n, locs1, locs2, mass_pre, mass_t, mass_nxt, dqi, dqimax, dqimin) firstprivate(ob, nloc)
      do ot = 1, nloc
         do inner = 1, inner_size
            do i = 1, 3*lonn
               og = ob + ot - 1
               if (nstep_less(inner, og)) then
                  if (i .ge. kstr(inner, ot) - 2 .and. i .le. kend(inner, ot) + 2) then
                     imp = outer_index(2, og)
                     call locs_pp_map(pp(1, inner, og), i, lonn, imp, sc, locs1)
                     call locs_pp_map(pp(1, inner, og), i + 1, lonn, imp, sc, locs2)
                     hh(i, inner, ot) = locs2 - locs1
                     do n = 1, nvars
                        mass_pre = qq(mod(i - 2, imp) + 1, n, inner, og)
                        mass_t = qq(mod(i - 1, imp) + 1, n, inner, og)
                        mass_nxt = qq(mod(i, imp) + 1, n, inner, og)
                        dqi = 0.25*(mass_nxt - mass_pre)
                        dqimax = max(mass_pre, mass_t, mass_nxt) - mass_t
                        dqimin = mass_t - min(mass_pre, mass_t, mass_nxt)
                        dqmono(i, n, inner, ot) = sign(min(abs(dqi), dqimin, dqimax), dqi)
                     end do
                  end if
               end if
            end do
         end do
      end do

      !$omp target teams distribute parallel do collapse(3) private(og, imp, n, locs1, locs2, locs3, hh1, hh2, cc, mass_pre, mass_t, dqmono_pre, dqmono_cur) firstprivate(ob, nloc)
      do ot = 1, nloc
         do inner = 1, inner_size
            do i = 1, 3*lonn
               og = ob + ot - 1
               if (nstep_less(inner, og)) then
                  if (i .ge. kstr(inner, ot) - 1 .and. i .le. kend(inner, ot) + 2) then
                     imp = outer_index(2, og)
                     call locs_pp_map(pp(1, inner, og), i - 1, lonn, imp, sc, locs1)
                     call locs_pp_map(pp(1, inner, og), i, lonn, imp, sc, locs2)
                     call locs_pp_map(pp(1, inner, og), i + 1, lonn, imp, sc, locs3)
                     hh1 = locs3 - locs2
                     hh2 = locs2 - locs1
                     cc = 1./(hh1 + hh2)
                     hh1 = hh1*cc
                     hh2 = hh2*cc
                     do n = 1, nvars
                        mass_pre = qq(mod(i - 2, imp) + 1, n, inner, og)
                        mass_t = qq(mod(i - 1, imp) + 1, n, inner, og)
                        dqmono_pre = dqmono(i - 1, n, inner, ot)
                        dqmono_cur = dqmono(i, n, inner, ot)
                        qi(i, n, inner, ot) = mass_pre*hh1 + mass_t*hh2 + (dqmono_pre - dqmono_cur)/3.
                     end do
                  end if
               end if
            end do
         end do
      end do

      !$omp target teams distribute parallel do collapse(3) private(og, imp, imf, n, kkh, kkn, locs1, th, th2, th3, thp, thm, thc, inside, accumulative, qmi_t, qpi_t, mass_t, c1, c2, cc, dql_t, dqq, kk) firstprivate(ob, nloc)
      do ot = 1, nloc
         do inner = 1, inner_size
            do i = 1, lonn + 1
               og = ob + ot - 1
               if (nstep_less(inner, og)) then
                  imp = outer_index(2, og)
                  imf = outer_index(3, og)
                  if (i .le. imf + 1) then
                     kkh = kkh_array(i, inner, ot)
                     kkn = kkh
                     if (i .ne. imf + 1) then
                        kkn = kkh_array(i + 1, inner, ot)
                     end if
                     accumulative = (i .ne. imf + 1) .and. (kkh .ne. kkn)
                     call locs_pp_map(pp(1, inner, og), kkh, lonn, imp, sc, locs1)
                     th = (pn(i, inner, og) - locs1)/hh(kkh, inner, ot)
                     tl_array(i, inner, ot) = th
                     th2 = th*th
                     th3 = th2*th
                     thp = th3 - th2
                     thm = th3 - 2.*th2 + th
                     thc = -2.*th3 + 3.*th2
                     inside = (mono .eq. 1) .and. (kkh .ge. kstr(inner, ot) - 1) .and. (kkh .le. kend(inner, ot) + 1)
                     do n = 1, nvars
                        qmi_t = 0.0
                        qpi_t = 0.0
                        mass_t = qq(mod(kkh - 1, imp) + 1, n, inner, og)
                        if (inside) then
                           qmi_t = qi(kkh, n, inner, ot)
                           qpi_t = qi(kkh + 1, n, inner, ot)
                           c1 = qpi_t - mass_t
                           c2 = mass_t - qmi_t
                           if (c1*c2 .le. 0.0) then
                              qmi_t = mass_t
                              qpi_t = mass_t
                           else
                              cc = qpi_t - qmi_t
                              c1 = cc*(mass_t - 0.5*(qpi_t + qmi_t))
                              c2 = cc*cc/6.
                              if (c1 .gt. c2) then
                                 qmi_t = 3.*mass_t - 2.*qpi_t
                              else if (c1 .lt. -c2) then
                                 qpi_t = 3.*mass_t - 2.*qmi_t
                              end if
                           end if
                        end if
                        dql_t = thp*qpi_t + thm*qmi_t + thc*mass_t
                        dql_array(i, n, inner, ot) = dql_t
                        if (accumulative) then
                           dqq = (mass_t - dql_t)*hh(kkh, inner, ot)
                           do kk = kkh + 1, kkn - 1
                              dqq = dqq + qq(mod(kk - 1, imp) + 1, n, inner, og)*hh(kk, inner, ot)
                           end do
                           dqq_array(i, n, inner, ot) = dqq
                        end if
                     end do
                  end if
               end if
            end do
         end do
      end do

      !$omp target teams distribute parallel do collapse(3) private(og, imp, imf, n, dpp, kkl, kkh, tl, th, dql_t, dqh_t, rdthtl, qn_t, dqq, kk) firstprivate(ob, nloc)
      do ot = 1, nloc
         do inner = 1, inner_size
            do i = 1, lonn
               og = ob + ot - 1
               if (nstep_less(inner, og)) then
                  imp = outer_index(2, og)
                  imf = outer_index(3, og)
                  if (i .le. imf) then
                     dpp = 1.0
                     kkl = kkh_array(i, inner, ot)
                     kkh = kkh_array(i + 1, inner, ot)
                     tl = tl_array(i, inner, ot)
                     th = tl_array(i + 1, inner, ot)
                     if (kkh .gt. kkl) then
                        dpp = (1.0 - tl)*hh(kkl, inner, ot) + th*hh(kkh, inner, ot)
                        do kk = kkl + 1, kkh - 1
                           dpp = dpp + hh(kk, inner, ot)
                        end do
                     end if
                     do n = 1, nvars
                        dql_t = dql_array(i, n, inner, ot)
                        dqh_t = dql_array(i + 1, n, inner, ot)
                        if (kkh .eq. kkl) then
                           rdthtl = th - tl
                           if (rdthtl .ne. 0.) rdthtl = 1./rdthtl
                           qn_t = (dqh_t - dql_t)*rdthtl
                        else
                           dqq = dqq_array(i, n, inner, ot) + dqh_t*hh(kkh, inner, ot)
                           qn_t = dqq/dpp
                        end if
                        qq(i, n, inner, og) = qn_t
                     end do
                  end if
               end if
            end do
         end do
      end do
   end do

   ! #region agent log
   dbg0 = int(lonn, 8); dbg1 = int(nvars, 8); dbg2 = int(otile, 8)
   call geps_dbg_vram(1, 718, dbg0, dbg1, dbg2)
   ! #endregion

   !$omp target exit data map(delete:kstr, kend, kkh_array, hh, dqmono, qi, tl_array, dql_array, dqq_array) map(from:has_error)
   if (has_error) then
      print *, "[ERROR] There is an error in cyclic_cell_ppm_intp_two_loops_gpu."
      print *, "[ERROR] Please check pp and pn."
      call abort
   end if
end subroutine cyclic_cell_ppm_intp_two_loops_gpu
