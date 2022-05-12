import random

from visdom import Visdom


def draw_acc_loss(acc_list, loss_list, vis: Visdom, line_id):

    for idx, (acc, loss) in enumerate(zip(acc_list, loss_list)):
        vis.line([acc], [idx], win='acc', name='line' + str(line_id) + '_acc', update='append')
        vis.line([loss], [idx], win='loss', name='line' + str(line_id) + '_loss', update='append')

