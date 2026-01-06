#!/usr/bin/env python
from datetime import datetime
import os
from utilgpu import pick_gpu_lowest_memory
from Engine import *

optobj = Options()
optobj.parse_args(*sys.argv[1:])
optobj.process_options()

# if device number not set
if optobj.opts['device'] == "cuda":
    if torch.cuda.device_count() > 1:
        gid, _ = pick_gpu_lowest_memory()
        optobj.opts['device'] = f"cuda:{gid}"
    

pid = os.getpid()
host_name = os.popen('hostname').read().strip()
# save command to file
f = open("commands.txt", "a")
f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
f.write(f'  pid: {pid}')
f.write(f'  host: {host_name}')
f.write('\n')
f.write(' '.join(sys.argv))
f.write('\n')
f.close()


eng = Engine(optobj.opts)
eng.run()
# eng.pde.visualize(savedir=eng.logger.get_dir())
