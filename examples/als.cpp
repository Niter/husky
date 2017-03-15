// Copyright 2015 Husky Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "boost/tokenizer.hpp"
#include <Eigen/Dense>

#include "core/executor.hpp"
#include "core/objlist.hpp"
#include "core/utils.hpp"
#include "io/input/inputformat_store.hpp"
#include "io/input/line_inputformat.hpp"
#include "io/hdfs_manager.hpp"
#include "lib/aggregator_factory.hpp"

#include "lib/vector.hpp"

typedef Eigen::VectorXd VectorT;
typedef Eigen::MatrixXd MatrixT;

const int MAGIC = 50000000;
typedef Eigen::VectorXd VectorT;
typedef Eigen::MatrixXd MatrixT;
typedef std::pair<double, VectorT> FactorMsgT;

/// Any class that subclasses it can be able to do ALS
/// TODO need some machanism to "remember" the deduplicated requests
class ALSNode{
public:
    ALSNode() {}
    ALSNode(int id) : key(id) {
        ever_active = false;
    }

    static void set_rank(int _rank) {
        rank = _rank;
    }
    static void set_lambda(double _lambda) {
        lambda = _lambda;
    }
    static void set_iter(int _iter) {
        iter = _iter;
    }

    typedef int KeyT;
    KeyT key;
    bool active;
    bool ever_active;
    std::vector<int> nbs;
    std::vector<double> obs;
    VectorT factors;

    static int rank;  // Default is 20
    static double lambda;  // Default is 0.01
    static int iter;  // Default is 10

    virtual KeyT const & id() const {
        return key;
    }

    void broadcast(husky::PushChannel<FactorMsgT, ALSNode>& ch) {
        ever_active = true;
        for(int i = 0; i < nbs.size(); i++) {
            ch.push(FactorMsgT(obs[i], factors), nbs[i]);
        }
    }

    friend husky::BinStream& operator<<(husky::BinStream& stream, const ALSNode& node) {
        stream << node.key << node.active << node.ever_active << node.nbs << node.obs << node.factors;
        return stream;
    }

    friend husky::BinStream& operator>>(husky::BinStream& stream, ALSNode& node) {
        stream >> node.key >> node.active >> node.ever_active >> node.nbs >> node.obs >> node.factors;
        return stream;
    }
};

int ALSNode::rank = 20;
double ALSNode::lambda = 0.01;
int ALSNode::iter = 10;

class UserItemRatingObject {
   public:
    using KeyT = int;
    explicit UserItemRatingObject(int _user, int _item, double _rating) : user(_user), item(_item), rating(_rating) {
		std::mt19937 rng;
		rng.seed(std::random_device()());
		std::uniform_int_distribution<std::mt19937::result_type> dist(1, 1 << 31);
		objid = dist(rng);
    }
    KeyT objid;
    int user;
    int item;
    double rating;
    const KeyT& id() const { return objid; }
};


// TODO: Serialization for VectorT (Eigen)
void als() {
    // Prepare ObjList for later ALS
    int num_worker = husky::Context::get_num_workers();
    int worker_id = husky::Context::get_global_tid();
    auto& als_list = husky::ObjListStore::create_objlist<ALSNode>("als-node");
    auto& ac = husky::lib::AggregatorFactory::get_channel();
    // Parse Data
    auto& tmp_store = husky::ObjListStore::create_objlist<UserItemRatingObject>("tmp");  // Intented to be stayed at local
    husky::lib::Aggregator<int> max_user_index_agg(0, [](int&a, const int& b) {a = std::max(a, b);});
    husky::lib::Aggregator<int> max_item_index_agg(0, [](int&a, const int& b) {a = std::max(a, b);});
    husky::lib::Aggregator<int> num_rating_agg(0, [](int&a, const int& b) {a += b;});
    auto& infmt = husky::io::InputFormatStore::create_line_inputformat();
    std::string url = husky::Context::get_param("input");
    husky::LOG_I << url;
    infmt.set_input(url);
    std::function<void(boost::string_ref)> parser = [&](boost::string_ref chunk) {
        if (chunk.empty())
            return;
        boost::char_separator<char> sep(" \t");
        boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);

        auto it = tok.begin();
        int user = std::stoi(*it);
        it++;
        int item = std::stoi(*it)+MAGIC;
        it++;
        double rating = std::stof(*it);

        max_user_index_agg.update(user);
        max_item_index_agg.update(item);
        tmp_store.add_object(UserItemRatingObject(user, item, rating));
    };
    husky::load(infmt, {&ac}, parser);
    husky::LOG_I << "Finished Loading From HDFS";
    // loadData into ALS Obj
    int num_node = std::max(max_item_index_agg.get_value(), max_user_index_agg.get_value());
    husky::LOG_I << "num_node: " << std::to_string(num_node);
    for (int i = worker_id; i <= num_node; i += num_worker) {
        als_list.add_object(ALSNode(i));
    }
    husky::LOG_I << "Created als_list";
    husky::globalize(als_list);
    husky::LOG_I << "Balanced als_list";
    husky::PushChannel<std::pair<int, double>, ALSNode>& data_push_channel = 
        husky::ChannelStore::create_push_channel<std::pair<int, double>>(tmp_store, als_list);
    husky::list_execute(tmp_store, {}, {&data_push_channel}, [&](UserItemRatingObject& obj) {
        data_push_channel.push(std::pair<int, double>(obj.item, obj.rating), obj.user);
        data_push_channel.push(std::pair<int, double>(obj.user, obj.rating), obj.item);
    });
    husky::LOG_I << "Pushed Data from tmp_store to als_list";
    husky::list_execute(als_list, {&data_push_channel}, {}, [&](ALSNode& node){
        auto& vec_msg = data_push_channel.get(node);
        for (auto& msg : vec_msg) {
            node.nbs.push_back(msg.first);
            node.obs.push_back(msg.second);
        }
    });
    husky::LOG_I << "als_list received data";
    husky::list_execute(als_list, {}, {}, [&](ALSNode& node){
        if (node.nbs.size() == 0) als_list.delete_object(&node);
    });
    husky::LOG_I << "deleted useless obj in als_list";
    husky::balance(als_list);
    husky::LOG_I << "blanced after deletion in als_list";
    // Set the parameter for training
    int num_iter = 10;
    // Initiate ALS list
    auto& factors_push_channel = husky::ChannelStore::create_push_channel<FactorMsgT>(als_list, als_list);
    husky::list_execute(als_list, {}, {&factors_push_channel}, [&](ALSNode& node) {
		node.factors.resize(node.rank);
		node.factors.setRandom();
		if(node.key < MAGIC) {
			node.broadcast(factors_push_channel);
			node.active = false;
		} else {
			node.active = true;
		}
		return;
    });
    husky::LOG_I << "Initilizated als_list";
    // Train ALS
    for (int iter_train = 0; iter_train < num_iter; iter_train++) {
        husky::list_execute(als_list, {&factors_push_channel}, {&factors_push_channel}, [&] (ALSNode & node) {
            // If I'm type 0 and I'm initing, send msg
            if(node.active == false) {
                node.active = true;
                return;
            } else {
                MatrixT sum_mat;
                VectorT sum_vec = VectorT::Zero(node.rank);
                auto & recv_data = factors_push_channel.get(node);
                if (recv_data.size() == 0) return;
                for(int i = 0; i < recv_data.size(); i++) {
                    double rating = recv_data[i].first;
                    auto & other_factors = recv_data[i].second;
                    assert(other_factors.size() == node.rank);
                    if(sum_mat.size() == 0) {
                        sum_mat.resize(node.rank, node.rank);
                        sum_mat.triangularView<Eigen::Upper>() = other_factors * other_factors.transpose();
                    } else {
                        sum_mat.triangularView<Eigen::Upper>() += other_factors * other_factors.transpose();
                    }
                    sum_vec += other_factors * rating;
                }
                // husky::LOG_I << "regularization";
                double regularization = node.lambda*node.nbs.size()*(node.key < MAGIC);
                for(int i = 0; i < sum_mat.rows(); ++i) 
                    sum_mat(i,i) += regularization; 
                // TODO
                // husky::LOG_I << "added regularization";
                node.factors = sum_mat.selfadjointView<Eigen::Upper>().ldlt().solve(sum_vec);
                node.broadcast(factors_push_channel);
                node.active = false;
            }
        });
    }
    husky::LOG_I << "Trained als_list";
    // Show The result, get the rmse
    auto rmse_agg = husky::lib::Aggregator<double>(0.0, [](double& a, const double& b){ a += b; });
    husky::list_execute(als_list, {&factors_push_channel}, {&ac}, [&](ALSNode& node){
		if(not node.active) return;
		MatrixT sum_mat;
		VectorT sum_vec = VectorT::Zero(node.rank);
        auto & recv_data = factors_push_channel.get(node);
        if (recv_data.size() == 0) return;
        for(int i=0; i<recv_data.size(); i++) {
            double rating = recv_data[i].first;
            auto & X = recv_data[i].second;
			assert(X.size() == node.rank);
			MatrixT XtX(node.rank, node.rank);
			if(sum_mat.size() == 0) {
				sum_mat.resize(node.rank, node.rank);
				sum_mat.triangularView<Eigen::Upper>() = X * X.transpose();
			} else {
				sum_mat.triangularView<Eigen::Upper>() += X * X.transpose();
			}
			sum_vec += X * rating;
		}
		node.factors = sum_mat.selfadjointView<Eigen::Upper>().ldlt().solve(sum_vec);

		for(int i = 0; i < recv_data.size(); i++) {
            double rating = recv_data[i].first;
            auto & X = recv_data[i].second;
			double pred = X.dot(node.factors);
			pred = std::max(1., pred);
			pred = std::min(5., pred);
			double loss = pred - rating;
			loss *= loss;
			rmse_agg.update(loss);
            num_rating_agg.update(1);
		}
    });
    int num_rating = num_rating_agg.get_value();
	if (worker_id == 0) {
		double rmse = rmse_agg.get_value();
		rmse = sqrt(rmse/num_rating);
        husky::LOG_I << "num_rating: " << std::to_string(num_rating);
        husky::LOG_I << std::to_string(rmse);
	}
}

int main(int argc, char** argv) {
    std::vector<std::string> args;
    args.push_back("hdfs_namenode");
    args.push_back("hdfs_namenode_port");
    args.push_back("input");
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(als);
        return 0;
    }
    return 1;
}
