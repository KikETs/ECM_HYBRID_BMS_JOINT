from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/"firmware"
shutil.copytree(ROOT.parent/"repo"/"mcu",R,symlinks=True,ignore=shutil.ignore_patterns("Build","__pycache__"))

def inject(path, old, new):
    p=R/path
    text=p.read_text()
    if text.count(old)!=1: raise RuntimeError((path,text.count(old),old[:50]))
    p.write_text(text.replace(old,new))

stress="""
float sop_review_stress(sop_dir_t dir, float soc, float soh, float vpre,
                        float cutoff, float tau, float kf, float ks)
{
    volatile unsigned masks = 0U;
    volatile float audit_current = 0.0f;
    float trial = dir == SOP_CHARGE ? 5.0f : -12.0f;
    for (unsigned n = 0; n < SOP_MAX_ITER; ++n)
    {
        masks += (unsigned)review_in_hull(dir,soc,soh,trial);
        float r = sop_r_eff(dir,soc,soh,trial,tau,kf,ks);
        if (!isfinite(r) || !(r > 0.0f)) r = 0.01f;
        float next = (cutoff-vpre)/r;
        if (!isfinite(next)) next = trial;
        if (dir == SOP_CHARGE) next = fminf(400.0f,fmaxf(0.05f,next));
        else next = fminf(-0.1f,fmaxf(-400.0f,next));
        masks += (unsigned)review_in_hull(dir,soc,soh,next);
        audit_current = fabsf(next-trial);
        trial = 0.5f*next+0.5f*trial;
    }
    (void)audit_current;
    (void)masks;
    return trial;
}
"""
inject("sop_core.c","/* ---- 표 조회 보조:",stress+"\n/* ---- 표 조회 보조:")
inject("sop_core.h","\n#endif\n","""
float sop_review_stress(sop_dir_t dir, float soc, float soh, float vpre,
                       float cutoff, float tau, float kf, float ks);
#endif
""")
inject("fw_sop/Src/main.c","#define SOP_CMD_REVIEW4  0x72U","#define SOP_CMD_REVIEW4  0x72U\n#define SOP_CMD_WCET     0x73U")
inject("fw_sop/Src/main.c","&& cmd != SOP_CMD_PACK && cmd != SOP_CMD_REVIEW4","&& cmd != SOP_CMD_PACK && cmd != SOP_CMD_REVIEW4 && cmd != SOP_CMD_WCET")
inject("fw_sop/Src/main.c","|| command == SOP_CMD_PACK || command == SOP_CMD_REVIEW4","|| command == SOP_CMD_PACK || command == SOP_CMD_REVIEW4 || command == SOP_CMD_WCET")
inject("fw_sop/Src/main.c","  if (cmd == SOP_CMD_REVIEW4 && (q.dir & 0x80U))", """
  if (cmd == SOP_CMD_WCET)
  {
    unsigned index = q.dir & 7U;
    if (index >= SOP_PACK_NSET) index = SOP_PACK_NSET - 1U;
    n_cells = SOP_PACK_N[index];
    if (q.dir & 0x40U)
    {
      if ((ICACHE->CR & ICACHE_CR_EN) && HAL_ICACHE_Disable() != HAL_OK) Error_Handler();
    }
    else
    {
      if (!(ICACHE->CR & ICACHE_CR_EN) && HAL_ICACHE_Enable() != HAL_OK) Error_Handler();
      if ((q.dir & 0x20U) && HAL_ICACHE_Invalidate() != HAL_OK) Error_Handler();
    }
    for (unsigned c=0; c<n_cells; ++c)
    {
      float d = ((int)(c % 7U) - 3) * 0.001f;
      sop_feat_reset(&g_pack_feat[c]);
      sop_ekf_reset(&g_pack_ekf[c],q.soc+d,q.soh);
    }
  }
  if (cmd == SOP_CMD_REVIEW4 && (q.dir & 0x80U))""")
inject("fw_sop/Src/main.c","    case SOP_CMD_REVIEW4:", """
    case SOP_CMD_WCET:
    {
      float reduced[4] = {400.0f,400.0f,400.0f,400.0f};
      unsigned peak_iters=0, failures=0;
      int stress = (q.dir & 0x10U) != 0U;
      for (unsigned c=0; c<n_cells; ++c)
      {
        float voltage=q.v_pre+((int)(c%9U)-4)*0.001f;
        float soc=sop_ekf_step(&g_pack_ekf[c],1.0f,q.current_a,voltage,
                               q.soh,fabsf(q.current_a)<=1.0f);
        float x[SOP_NFEAT]={0.0f};
        (void)sop_feat_update_a8(&g_pack_feat[c],1.0f,q.current_a,voltage,soc,q.soh,x);
        for (unsigned d=0; d<2; ++d)
        {
          float kfast,kslow;
          sop_trim((sop_dir_t)d,x,&kfast,&kslow);
          for (unsigned h=0; h<2; ++h)
          {
            float limit=0.0f;
            float cutoff = d==SOP_CHARGE?4.2f:2.5f;
            float tau=h==0?2.0f:10.0f;
            uint32_t n;
            if (stress)
            {
              limit=sop_review_stress((sop_dir_t)d,soc,q.soh,voltage,cutoff,tau,kfast,kslow);
              n=24;
            }
            else
            {
              sop_review_status_t status=sop_review_solve((sop_dir_t)d,soc,q.soh,voltage,cutoff,
                                                        tau,kfast,kslow,&limit,&n);
              failures += status != SOP_REVIEW_OK;
            }
            if (n>peak_iters) peak_iters=n;
            reduced[d*2+h]=fminf(reduced[d*2+h],fabsf(limit));
          }
        }
      }
      it=(peak_iters<<16U)|failures;
      kf=reduced[0]; ks=reduced[1]; reff=reduced[2]; ist=reduced[3];
      break;
    }
    case SOP_CMD_REVIEW4:""")
