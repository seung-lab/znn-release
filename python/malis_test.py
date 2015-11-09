# test the malis of boundary map
import time
import emirt
#%% parameters
def get_params():
    pars = dict()
    # epsilone: a small number for log to avoind -infinity
    pars['eps'] = 0.0000001

    # largest disk radius
    pars['Dm'] = 500
    pars['Ds'] = 500

    # make a fake test image
    pars['is_fake'] = True

    # use aleks malis
    pars['is_aleks'] = False

    # whether using constrained malis
    pars['is_constrained'] = False

    # thicken boundary of label by morphological errosion
    pars['erosion_size'] = 0

    # a small corner
    pars['corner_size'] = 0

    # disk radius threshold
    pars['DrTh'] = 0
    return pars


def aleks_malis(bdm, lbl):
    print "use Aleks' implementation..."
    # prepare the binary files
    emirt.emio.znn_img_save(bdm.astype('float64'), '../dataset/bdm.image')
    emirt.emio.znn_img_save(lbl.astype('float64'), '../dataset/lbl.image')

    import os
    os.system('../bin/debug --options=../malis.config --type=malis_2d --high=1.0 --low=0.0')

    # read the output
    me = emirt.emio.znn_img_read('../experiments/out.merger')
    se = emirt.emio.znn_img_read('../experiments/out.splitter')

    # adjust the coordinate
    print "shape: ", me.shape
    print "maximum merger   weight: ", me.max()
    print "maximum splitter weight: ", se.max()
    me = me[0,0,:,:] + me[1,0,:,:]
    se = se[0,0,:,:] + se[1,0,:,:]
    me = me.reshape( bdm.shape )
    se = se.reshape( bdm.shape )

    w = me + se
    return w, me, se

def constrained_aleks_malis(bdm, lbl, threshold=0.5):
    """
    adding constraints for malis weight
    fill the intracellular space with ground truth when computing merging error
    fill the boundary with ground truth when computing spliting error
    """
    from cost_fn import constrain_label
    mbdm, sbdm = constrain_label(bdm, lbl)
    # get the merger weights
    mw, mme, mse = aleks_malis(mbdm, lbl)
    # get the splitter weights
    sw, sme, sse = aleks_malis(sbdm, lbl)
    w = mme + sse
    return (w, mme, sse)

if __name__ == "__main__":
    # get the parameters
    pars = get_params()

    import data_prepare
    bdm, lbl = data_prepare.read_image(pars)

    # recompile and use cost_fn
    #print "compile the cost function..."
    #os.system('python compile.py cost_fn')
    import cost_fn
    start = time.time()
    if pars['is_constrained']:
        if pars['is_aleks']:
            w, me, se = constrained_aleks_malis(bdm, lbl)
        else:
            print "compute the constrained malis weight..."
            w, me, se = cost_fn.constrained_malis_weight_bdm_2D(bdm, lbl)
    else:
        if pars['is_aleks']:
            print "normal malis with aleks version..."
            w, me, se = aleks_malis(bdm, lbl)
        else:
            print "normal malis weight with python version..."
            w, me, se = cost_fn.malis_weight_bdm_2D(bdm, lbl)

    elapsed = time.time() - start
    print "elapsed time is {} sec".format(elapsed)

    import malis_show
    malis_show.plot(pars, bdm, lbl, me, se)

    print "------end-----"
