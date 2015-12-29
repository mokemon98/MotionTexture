import numpy as np
import scipy.weave
import pylab as pl
import LDS
import sys
import time

class LdsManager:

    TAG = "[LDSM]"

    class MotionSegment:

        def __init__(self):
            self.start = 0
            self.end = 0
            self.label = 0

        def set(self, start, end, label):
            self.start = start
            self.end = end
            self.label = label

    def __init__(self, len_min=60, dim=12, thresh_micro=1.0, thresh_macro=1.0, framerate=60):
        self.LDSs = []
        self.lm = len_min
        self.dim = dim
        self.thresh_micro = thresh_micro
        self.thresh_macro = thresh_macro
        self.framerate = framerate
        self.segment = []

    def _error(self, data1, data2):
        e = (data1 - data2) ** 2
        # e_mean = np.mean(e, axis=0)
        # e_mean2 = np.mean(e_mean)
        e_max = np.max(e)
        return e_max

    def _one_error(self, data1, data2):
        e = (data1 - data2) ** 2
        #e_mean = np.mean(e)
        e_max = np.max(e)
        return e_max

    def _optimize_lds(self, data):

        if len(self.LDSs) == 0:
            return None, self.thresh_micro + 1

        error = []
        for lds in self.LDSs:
            pred = lds.predict(data)
            e = self._error(data, pred)
            error.append(e)
        error = np.array(error)
        e_min = np.min(error)
        idx = np.argmin(error)
        #lds = self.LDSs[idx]

        return idx, e_min

    def _init_train(self, start, end):
        first = self.data[start:end]
        lds = LDS.LDS(self.dim)
        lds.fit([first])

        first_pred = lds.predict(first)
        error = self._error(first, first_pred)

        if error > self.thresh_micro:
            print self.TAG, "Warning: fitting failed..."
            return lds, end

        error = 0
        test_end = end + 1

        while error < self.thresh_micro and test_end < len(self.data):
            test = self.data[start:test_end]
            test_pred = lds.predict(test)
            error = self._error(test, test_pred)
            test_end += 1

        return lds, test_end

    def _init_test(self, test):

        start = 0
        end = self.lm

        if end >= len(test):
            print self.TAG, "Test data is not enough"
            return 0

        error_list = []

        while end < len(test):

            label, e_min = self._optimize_lds(test[start:end])
            error_list.append(e_min)

            lds = self.LDSs[label]
            end += 1
            error = 0
            while error < self.thresh_micro and end < len(test):
                x = test[start:end]
                pred = lds.predict(x)
                error = self._error(x, pred)
                end += 5

            start = end
            end = start + self.lm

        e_mean = np.mean(np.array(error_list))

        return e_mean

    def _dp(self, error):

        T = len(self.data)
        N = (T - 1) / self.lm
        Tmin = self.lm
        Nt = len(self.LDSs)

        G = self.G
        E = self.E
        F = self.F

        error = np.array(error)

        code = """
            int N = (T - 1) / Tmin;

            for (int n=1; n<N; n++) {

                printf("\\r\\t\\t%d / %d", n, N);

                int TS = Tmin * (n + 1) - 1;
                int b_offset = n * Tmin;

                int TE;
                if (n < Nt - 1) {
                    TE = T - (Nt - n - 2) * Tmin;
                    TE = T;
                } else {
                    TE = T;
                }

                for (int t=TS; t<TE; t++) {

                    int b_range = (t - Tmin) - b_offset + 1;

                    if (b_range <= 0) {
                        G2(n, t) = G2(n-1, t);
                        E2(n, t) = E2(n-1, t);
                        F2(n, t) = F2(n-1, t);
                        continue;
                    }

                    double *L;
                    L = (double*)malloc(Nt * b_range * sizeof(double));

                    for (int i=0; i<Nt; i++) {

                        double e_sum = 0;
                        double e_max = 0;

                        for (int j=b_range+b_offset; j<t; j++) {
                            double e = ERROR2(i, j);
                            e_sum += e;

                            //if (e > e_max) {
                            //    e_max = e;
                            //}
                        }

                        for (int b=b_range-1; b>=0; b--) {

                            double g = G2(n-1, b+b_offset-1);

                            double e = ERROR2(i, b_offset+b);
                            e_sum += e;
                            //if (e > e_max) {
                            //    e_max = e;
                            //}

                            e = e_sum / (t - b_offset - b);

                            L[i*b_range+b] = g + e;

                        }

                    }

                    double min = 100000;
                    int min_i, min_b;

                    for (int i=0; i<Nt; i++) {
                        for (int b=0; b<b_range; b++) {
                            double li = L[i*b_range+b];
                            if (li < min) {
                                min = li;
                                min_i = i;
                                min_b = b;
                            }
                        }
                    }

                    G2(n, t) = min;
                    E2(n, t) = min_i;
                    F2(n, t) = min_b + b_offset;

                    free(L);
                }
            }

            printf("\\n");
        """

        if True:
            scipy.weave.inline(code, ["T", "Tmin", "Nt", "G", "E", "F", "error"])
        else:
            t1 = time.time()

            for n in xrange(1, N):

                # print "\t\t%d / %d" % (n+1, N)
                t2 = time.time()
                sys.stdout.write("\r\t\t%d / %d (%f)" % (n+1, N, t2-t1))
                sys.stdout.flush()

                TS = self.lm*(n+1) - 1
                b_offset = n * self.lm
                for t in xrange(TS, T):
                    b_range = (t - self.lm) - b_offset + 1
                    if b_range <= 0:
                        self.G[n, t] = self.G[n-1, t]
                        self.E[n, t] = self.E[n-1, t]
                        self.F[n, t] = self.F[n-1, t]
                        continue
                    g = self.G[n-1, b_offset-1:b_offset+b_range-1]
                    b = np.zeros((T, b_range))
                    for i in xrange(b_range):
                       b[b_offset+i:t, i] = 1.0 / (t - (b_offset+i))
                    likelihood = np.dot(error, b)
                    likelihood += g
                    li = np.min(likelihood)
                    idx = np.argmin(likelihood)
                    self.G[n, t] = li
                    self.E[n, t] = idx / b_range
                    self.F[n, t] = idx % b_range + b_offset
                t1 = t2

        """
        for n in range(1, N):
            print "\t%d / %d" % (n+1, N)
            for t in range(self.lm*(n+1) + 1, len(self.data)):
                b_offset = n * self.lm
                b_range = (t - self.lm) - b_offset
                likelihood = np.zeros((Nt, b_range))
                for i in range(Nt):
                    for b in range(b_range):
                        g = self.G[n-1, b+b_offset-1]
                        e_list = error[i]
                        e = e_list[b+b_offset:t]
                        e_sum = np.sum(e)
                        likelihood[i, b] = g + e_sum
                li = np.min(likelihood)
                idx = np.argmin(likelihood)
                self.G[n, t] = li
                self.E[n, t] = idx / b_range
                self.F[n, t] = idx % b_range + b_offset
        """

    def _e_step(self):

        print self.TAG, "Estep =============="

        T = len(self.data)
        Nt = len(self.LDSs)
        N = (T - 1) / self.lm

        self.G = np.zeros((N, T), dtype=np.float64)
        self.E = np.zeros((N, T), dtype=np.int8)
        self.F = np.zeros((N, T), dtype=np.int16)

        self.G += 100

        train = self.data
        error = []
        for lds in self.LDSs:
            pred = lds.predict(train)
            e = (train - pred) ** 2
            e_mean = np.mean(e, axis=1)
            e_max = np.max(e, axis=1)
            error.append(e_max)

        print "\tInitial Process"

        TS = self.lm - 1
        TE = T-Nt*self.lm

        for t in range(TS, T-1):
            e_t = []
            for i in range(Nt):
                e_list = error[i]
                e = e_list[:t]
                e_mean = np.mean(e)
                e_max = np.max(e)
                e_t.append(e_mean)
            e_t = np.array(e_t)
            self.G[0, t] = np.min(e_t)
            self.E[0, t] = np.argmin(e_t)

        print "\tDP Process"

        t1 = time.time()
        self._dp(error)
        t2 = time.time()
        print "\t\ttime =", (t2-t1) * 1000

        print "\tFinal Process"

        g_t = np.min(self.G[:, T-1])
        Ns = np.argmin(self.G[:, T-1]) + 1

        print "\t\tsegment =", Ns

        print "\tBacktrack Process"

        h = []
        l = []

        h.append(T)
        l.append(self.E[Ns-1, T-1])

        for n in xrange(Ns-1, 0, -1):
            hn = self.F[n, h[-1] - 1]
            h.append(hn)
            l.append(self.E[n-1, hn-1])

        h.append(0)

        h = h[::-1]
        l = l[::-1]

        self.segment = []
        for i in xrange(len(l)):
            seg = self.MotionSegment()
            seg.set(h[i], h[i+1], l[i])
            self.segment.append(seg)

        print "\t segment =", h
        print "\t label =", l

        print "============================"

    def _m_step(self):

        print self.TAG, "Mstep"

        Nt = len(self.LDSs)

        self.LDSs = []

        for i in range(Nt):
            train = [self.data[seg.start:seg.end] for seg in self.segment if seg.label == i]
            if len(train) != 0:
                lds = LDS.LDS(self.dim)
                lds.fit(train)
                self.LDSs.append(lds)

    def init_fit(self, data):

        self.data = data

        start = 0
        end = self.lm

        test_error = 0
        count = 0

        while end < len(data):

            print self.TAG, "Temp Result (", count, len(self.LDSs), ")"

            label, e = self._optimize_lds(data[start:end])

            if e < self.thresh_micro:
                lds = self.LDSs[label]
                end += 1
                error = 0
                while error < self.thresh_micro and end < len(data):
                    x = data[start:end]
                    pred = lds.predict(x)
                    error = self._error(x, pred)
                    end += 1
            else:
                print "\t*** New Model Training Start ***"
                lds, end = self._init_train(start, end)
                self.LDSs.append(lds)
                label = len(self.LDSs) - 1
                print "\t*** New Model Training Finish ***"

            s = self.MotionSegment()
            s.set(start, end, label)
            self.segment.append(s)

            # if end < len(data):
            #    test_error = self._init_test(data[end:])
            if end >= len(data):
                return

            print "\tFirst Error =", e
            print "\tEnd Position =", end
            print "\tTest Error =", test_error

            # if test_error < self.thresh_macro:
            #     return

            start = end
            end = start + self.lm
            count += 1

        # if test_error > self.thresh_macro:
        #     print self.TAG, "Error: lds optimization failed..."
        #     return -1
        # else:
        #     print self.TAG, "LDS optimization successed !!"
        #     return 0

    def fit(self):

        for i in xrange(3):

            self._e_step()
            self._m_step()

    def write_segment(self, path):
        seg_list = []
        seg_list.append([self.framerate, -1])
        for s in self.segment:
            seg_list.append([s.end, s.label])
        seg_list = np.array(seg_list)
        np.savetxt(path, seg_list, fmt="%d", delimiter=",")
