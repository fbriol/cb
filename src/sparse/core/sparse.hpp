#include <boost/functional/hash.hpp>
#include <unordered_map>
#include <tuple>
#include <memory>
#include <string>

namespace std {

template <typename... T>
struct hash<tuple<T...>> {
  size_t operator()(tuple<T...> const& arg) const noexcept {
    return boost::hash_value(arg);
  }
};

}  // namespace std

class Matrix {
 public:
  using Key = std::tuple<uint32_t, uint32_t>;
  using Map = std::unordered_map<Key, double>;

  Matrix() = default;

  auto set(const Key& key, const double x) -> void {
    const auto& _key = ji_ ? Matrix::swap_key(key) : key;
    i_ = std::max(std::get<0>(key), i_);
    j_ = std::max(std::get<1>(key), j_);
    data_->emplace(std::pair<Key, double>{_key, x});
  }

  auto get(const Key& key) const -> double {
    const auto& _key = ji_ ? Matrix::swap_key(key) : key;
    if (data_->count(_key) == 0) {
      auto i = std::get<0>(_key);
      if (i > i_) {
        throw std::range_error(
            "index " + std::to_string(i) + " is out of bounds for axis " +
            std::to_string(ji_ ? 1 : 0) + " with size " + std::to_string(i_));
      }
      auto j = std::get<1>(_key);
      if (j > j_) {
        throw std::range_error(
            "index " + std::to_string(j) + " is out of bounds for axis " +
            std::to_string(ji_ ? 0 : 1) + " with size " + std::to_string(j_));
      }
      return 0;
    }
    return (*data_)[_key];
  }

  auto shape() const -> Key {
    if (data_->empty()) {
      return {0, 0};
    }
    if (ji_) {
      return std::make_tuple(j_ + 1, i_ + 1);
    }
    return std::make_tuple(i_ + 1, j_ + 1);
  }

 private:
  static auto swap_key(const Key& key) -> Key {
    return std::make_tuple(std::get<1>(key), std::get<0>(key));
  }

  std::shared_ptr<Map> data_{new Map};
  uint32_t i_ = 0;
  uint32_t j_ = 0;
  bool ji_{false};
};
