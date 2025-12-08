#!/usr/bin/env python3
import argparse
from logging import getLogger
import os

from cryspy.start import cryspy_init, cryspy_restart
from cryspy.util.utility import set_logger, backup_cryspy, clean_cryspy
from cryspy.interface import select_code
from cryspy.job.ctrl_job import Ctrl_job
'''
  Using BO and EA method
'''

def search_flow(comm,mpi_rank,mpi_size,noprint,logger):
    if os.path.isfile('lock_cryspy'):
        logger.error('lock_cryspy file exists')
        raise SystemExit(1)
    else: 
        with open('lock_cryspy', 'w'):
            pass    # create vacant file

    # ---------- initialize
    if not os.path.isfile('cryspy.stat'):
        cryspy_init.initialize(comm=comm, mpi_rank=mpi_rank, mpi_size=mpi_size)
        os.remove('lock_cryspy')
        raise SystemExit()                  # 进入下一次循环，而不是退出
        # continue                           
    # ---------- restart
    else:
        # only stat and init_struc_data in rank0 are important
        rin, init_struc_data = cryspy_restart.restart(comm=comm, mpi_rank=mpi_rank, mpi_size=mpi_size)

    select_code.check_calc_files(rin)
    if rin.stop_chkpt == 1:                 # ---------- check point 1
        logger.info('Stop at check point 1')
        os.remove('lock_cryspy')
        raise SystemExit()

    # ---------- mkdir work/fin
    os.makedirs('work/fin', exist_ok=True)

    jobs = Ctrl_job(rin, init_struc_data)   # ---------- instantiate Ctrl_job class
    jobs.check_job()                        # ---------- check job status
    jobs.handle_job()                       # ---------- handle job

    # ---------- recheck for skip and done
    
    cnt_recheck = 0
    while jobs.id_queueing:
        logger.info(f'job id queueing: {jobs.id_queueing}')
        cnt_recheck += 1
        logger.info(f'\n\nrecheck {cnt_recheck}\n')
        jobs.check_job()
        jobs.handle_job()
    else:
        jobs.check_job()                  # ---------- check job status
        jobs.handle_job()                 # ---------- handle job
        
    if not (jobs.id_queueing or jobs.id_running): 
        if rin.algo in ['BO', 'LAQA', 'EA', 'EA-vc']:   # -- next selection or generation
            jobs.next_sg(noprint)
        else:                                           # ---------- for RS
            logger.info('\nDone all structures!')

    os.remove('lock_cryspy')                            # ---------- unlock


def main():
    '''主计算流'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--debug', help='debug', action='store_true')
    parser.add_argument('-n', '--noprint', help='not printing to the console', action='store_true')
    args = parser.parse_args()
    noprint = args.noprint

    # ########## MPI start
    # ---------- MPI
    comm,mpi_rank,mpi_size = None,0,1
    # ---------- logger
    set_logger(
        noprint=args.noprint,
        debug=args.debug,
        logfile='log_cryspy',
        errfile='err_cryspy',
        debugfile='debug_cryspy',
    )
    logger = getLogger('cryspy')

    # while True:   #########  main search loop
    # 主循环
    search_flow(comm,mpi_rank,mpi_size,noprint,logger)


if __name__ == '__main__':
    # ---------- main
    main()
