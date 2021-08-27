from utils import *
from .augmentation import *
from .data_helper import *

class FDDataset(Dataset):
    def __init__(self, mode, modality='color', fold_index='<NIL>', image_size=128, augment = None, balance = False, ROI=None, cross_test=False):
        super(FDDataset, self).__init__()
        print('fold: '+str(fold_index))
        print(modality)

        self.augment = augment
        self.mode       = mode
        self.modality = modality
        self.balance = balance
        self.ROI = ROI

        self.channels = 3
        self.train_image_path = TRN_IMGS_DIR
        self.val_image_path = VAL_IMGS_DIR
        self.test_image_path = TST_IMGS_DIR
        self.image_size = image_size
        self.fold_index = fold_index
        self.cross_test = cross_test
        self.set_mode(self.mode,self.fold_index)

    def set_mode(self, mode, fold_index):
        self.mode = mode
        self.fold_index = fold_index
        print(mode)
        print('fold index set: ', fold_index)

        if self.mode == 'test':
            self.test_list = load_test_list()
            self.num_data = len(self.test_list)
            self.dataset = self.test_list
            print('set dataset mode: test')
            self.cross_test = self.cross_test and self.mode =='test'
            self.non_zeros = NonZeroCrop()

        elif self.mode == 'val':
            self.val_list = load_val_list()
            self.num_data = len(self.val_list)
            self.dataset = self.val_list
            print('set dataset mode: val')

        elif self.mode == 'train':
            self.train_list = load_train_list()
            random.shuffle(self.train_list)
            self.num_data = len(self.train_list)
            print('set dataset mode: train')

            if self.balance:
                self.train_list = transform_balance(self.train_list)
            self.dataset = self.train_list
        print(self.num_data)


    def __getitem__(self, index):

        if self.fold_index is None:
            print('WRONG!!!!!!! fold index is NONE!!!!!!!!!!!!!!!!!')
            return

        if self.mode == 'train':
            if self.balance:
                if random.randint(0,1)==0:
                    tmp_list = self.train_list[0]
                else:
                    tmp_list = self.train_list[1]

                pos = random.randint(0,len(tmp_list)-1)
                color, depth, ir, label = tmp_list[pos]
            else:
                color, depth, ir, label = self.train_list[index]
            DATA_ROOT = self.train_image_path
        elif self.mode == 'val':
            color, depth, ir, label = self.val_list[index]
            DATA_ROOT = self.val_image_path
            
        elif self.mode == 'test':
            color, depth, ir, label = self.test_list[index]
            # test_id = color+' '+depth+' '+ir
            DATA_ROOT = self.test_image_path
        # print(os.path.join(DATA_ROOT, depth))
        # print(os.path.join(DATA_ROOT, color))
        # print(os.path.join(DATA_ROOT, ir))

        color = cv2.imread(os.path.join(DATA_ROOT, color),1)
        depth = cv2.imread(os.path.join(DATA_ROOT, depth),1)
        ir = cv2.imread(os.path.join(DATA_ROOT, ir),1)
        # print('shape depth ', depth.shape)
        # print('shape ir ', ir.shape)
        # print('shape color ', color.shape)

        if self.cross_test:
            color = self.non_zeros(color)
            depth = self.non_zeros(depth)
            ir = self.non_zeros(ir)
        color = cv2.resize(color,(RESIZE_SIZE,RESIZE_SIZE))
        depth = cv2.resize(depth,(RESIZE_SIZE,RESIZE_SIZE))
        ir = cv2.resize(ir,(RESIZE_SIZE,RESIZE_SIZE))
            
        if self.mode == 'train':
            ROI = self.ROI
            if ROI is not None:
                roi_x = random.randint(0, RESIZE_SIZE-self.image_size)
                roi_y = random.randint(0, RESIZE_SIZE-self.image_size)
                ROI = (roi_x, roi_y)
            color = color_augumentor(color,target_shape=(self.image_size, self.image_size, 3), roi=ROI)
            depth = depth_augumentor(depth,target_shape=(self.image_size, self.image_size, 3), roi=ROI)
            ir = ir_augumentor(ir,target_shape=(self.image_size, self.image_size, 3), roi=ROI)
            try:
                color = cv2.resize(color, (self.image_size, self.image_size))
                depth = cv2.resize(depth, (self.image_size, self.image_size))
                ir = cv2.resize(ir, (self.image_size, self.image_size))
            except Exception as e:
                print(e)
                print('depth:\t', depth.shape)
                print('ir:\t', ir.shape)

            image = np.concatenate([color.reshape([self.image_size, self.image_size, 3]),
                                    depth.reshape([self.image_size, self.image_size, 3]),
                                    ir.reshape([self.image_size, self.image_size, 3])],
                                   axis=2)

            if random.randint(0, 1) == 0:
                random_pos = random.randint(0, 2)
                if random.randint(0, 1) == 0:
                    image[:, :, 3 * random_pos:3 * (random_pos + 1)] = 0
                else:
                    for i in range(3):
                        if i != random_pos:
                            image[:, :, 3 * i:3 * (i + 1)] = 0

            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image.reshape([self.channels * 3, self.image_size, self.image_size])
            image = image / 255.0

            label = int(label)
            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))

        elif self.mode == 'val':
            color = color_augumentor(color, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            depth = depth_augumentor(depth, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            ir = ir_augumentor(ir, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            n = len(color)

            color = np.concatenate(color, axis=0)
            depth = np.concatenate(depth, axis=0)
            ir = np.concatenate(ir, axis=0)

            image = np.concatenate([color.reshape([n,self.image_size, self.image_size, 3]),
                                    depth.reshape([n,self.image_size, self.image_size, 3]),
                                    ir.reshape([n,self.image_size, self.image_size, 3])],
                                   axis=3)

            image = np.transpose(image, (0, 3, 1, 2))
            image = image.astype(np.float32)
            image = image.reshape([n, self.channels * 3, self.image_size, self.image_size])
            image = image / 255.0

            label = int(label)
            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))


        elif self.mode == 'test':
            color = color_augumentor(color, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            depth = depth_augumentor(depth, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            ir = ir_augumentor(ir, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            n = len(color)

            color = np.concatenate(color, axis=0)
            depth = np.concatenate(depth, axis=0)
            ir = np.concatenate(ir, axis=0)

            image = np.concatenate([color.reshape([n, self.image_size, self.image_size, 3]),
                                    depth.reshape([n, self.image_size, self.image_size, 3]),
                                    ir.reshape([n, self.image_size, self.image_size, 3])],
                                   axis=3)

            image = np.transpose(image, (0, 3, 1, 2))
            image = image.astype(np.float32)
            image = image.reshape([n, self.channels * 3, self.image_size, self.image_size])
            image = image / 255.0
            
            label = int(label)
            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))

    def __len__(self):
        return self.num_data

    def analyze(self):
        d = {
            'real': 0,
            'fake': 0,
        }
        if self.balance:
            d['real'] = len(self.dataset[0])
            d['fake'] = len(self.dataset[1])
        else:
            for p in self.dataset:
                label = p[-1]
                if int(label) == 0:
                    d['fake'] += 1
                else:
                    d['real'] += 1
        return d


class NonZeroCrop(object):
    """Cut out black regions.
    """

    def __call__(self, img):
        arr = np.asarray(img)
        pixels = np.transpose(arr.nonzero())
        if len(arr.shape) > 2:
            pixels = pixels[:, :-1]
        top = pixels.min(axis=0)
        h, w = pixels.max(axis=0) - top
        return F.crop(img, top[0], top[1], h, w)


# check #################################################################
def run_check_train_data():
    dataset = FDDataset(mode = 'train')
    print(dataset)

    num = len(dataset)
    for m in range(num):
        i = np.random.choice(num)
        image, label = dataset[m]
        print(image.shape)
        print(label)

        if m > 100:
            break

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_check_train_data()


